import os
import shutil
import io
import tornado
from tornado.web import Application, RequestHandler, StaticFileHandler
from tornado.ioloop import IOLoop
import yaml
import json
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper
import PIL.Image
from aiortc import RTCPeerConnection, RTCSessionDescription
from tqdm import tqdm

from dnnlib.util import EasyDict as edict
from dataset_tool import open_dest
from nle_renderer import Renderer
from nle_util import to_json

renderer = Renderer()

CACHE_DIR = 'nle_cache'
CONFIG_PATH = 'models.yml'

class Handler(RequestHandler):
    def __init__(self, *args, models=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.set_header('Access-Control-Allow-Origin', '*')
        self.set_header('Access-Control-Allow-Headers', 'origin, content-type, accept, x-requested-with')
        self.set_header('Access-Control-Allow-Methods', 'POST, GET, OPTIONS')
        self.models = models

class Root(Handler):
    def get(self):
        self.write({'models': self.models})

class Latents(Handler):
    def get(self, **kwargs):
        model = kwargs.get('model')
        if 'latents' in self.models[model]:
            with open(self.models[model]['latents']) as f:
                latents = json.loads(f.read())
        else:
            samples_path = os.path.join(CACHE_DIR, model, 'samples')
            latents_path = samples_path + '_latents.json'
            with open(CONFIG_PATH, 'w') as f:
                models = dict(**self.models)
                models[model]['latents'] = latents_path
                models[model]['samples'] = samples_path
                f.write(yaml.dump(models))
            archive_root_dir, save_bytes, close_dest = open_dest(samples_path)
            latents = dict()
            for seed in tqdm(range(16384)):
                idx_str = f'{seed:08d}'
                archive_fname = f'{idx_str[:5]}/img{idx_str}.png'
                renderer.set_args(pkl=self.models[model]['network'], w0_seeds=[[seed, 1]])
                result = renderer.get()
                if 'image' in result:
                    img = result.image
                    img = PIL.Image.fromarray(img)
                    image_bits = io.BytesIO()
                    img.save(image_bits, format='png', compress_level=0, optimize=False)
                    save_bytes(os.path.join(archive_root_dir, archive_fname), image_bits.getbuffer())
                    latents[archive_fname] = {key: value for key, value in result.items() if key in ['all_cs', 'all_zs', 'all_ws', 'w0_seeds']}
                else:
                    print(result)
            with open(latents_path, 'w') as f:
                f.write(to_json(latents))
            close_dest()
        self.write(to_json(dict(**kwargs, latents=latents)))

class Interpolation(Handler):
    def get(self, **kwargs):
        model = kwargs.get('model')
        seeds = kwargs.get('seeds')
        self.write(self.request.body)
        return
        samples_path = os.path.join(CACHE_DIR, model, 'samples')
        latents_path = samples_path + '_latents.json'
        with open(CONFIG_PATH, 'w') as f:
            models = dict(**self.models)
            models[model]['latents'] = latents_path
            models[model]['samples'] = samples_path
            f.write(yaml.dump(models))
        archive_root_dir, save_bytes, close_dest = open_dest(samples_path)
        latents = dict()
        for seed in tqdm(range(10000)):
            idx_str = f'{seed:08d}'
            archive_fname = f'{idx_str[:5]}/img{idx_str}.png'
            renderer.set_args(pkl=self.models[model]['network'], w0_seeds=[[seed, 1]])
            result = renderer.get()
            img = result.image
            img = PIL.Image.fromarray(img)
            image_bits = io.BytesIO()
            img.save(image_bits, format='png', compress_level=0, optimize=False)
            save_bytes(os.path.join(archive_root_dir, archive_fname), image_bits.getbuffer())
            print(result)
            latents[archive_fname] = {
                    key: value for key, value in result.items()
                    if key in ['all_cs', 'all_zs', 'all_ws', 'w0_seeds']
                    }
        with open(latents_path, 'w') as f:
            f.write(to_json(latents))
        close_dest()
        self.write(to_json(dict(**kwargs, latents=latents)))


class Preview(Handler):
    def get(self, **kwargs):
        pass


generator = None


from nle_renderer.aiortc_renderer import Generator

class Offer(Handler):

    async def options(self):
        print('options', self.request.body)
        pass
    
    async def post(self):
        params = tornado.escape.json_decode(self.request.body)
        offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])
        pc = RTCPeerConnection()

        @pc.on("connectionstatechange")
        async def on_connectionstatechange():
            print("Connection state is %s" % pc.connectionState)
            if pc.connectionState == "failed":
                await pc.close()

        @pc.on("datachannel")
        def on_datachannel(channel):
            @channel.on("message")
            async def on_message(message):
                if isinstance(message, str):
                    data = json.loads(message)
                    self.generator.data(data)

        # open media source
        self.generator = Generator()
        video = self.generator.video

        await pc.setRemoteDescription(offer)
        for t in pc.getTransceivers():
            if t.kind == "video" and video:
                pc.addTrack(video)

        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)

        return self.write({"sdp": pc.localDescription.sdp, "type": pc.localDescription.type})

def make_app():
    with open(CONFIG_PATH) as f:
        models = edict(yaml.load(f, Loader=Loader))
    urls = [
            ('/', Root, {'models': models}),
            (r'/latents/(?P<model>\w+)', Latents, {'models': models}),
            (r'/interpolation/(?P<model>\w+)', Interpolation, {'models': models}),
            (r'/offer', Offer),
            *[
                (
                    r'/samples/%s/(.*)' % model,
                    StaticFileHandler,
                    {'path': os.path.abspath(os.path.join(CACHE_DIR, model, 'samples'))}
                )
                for model, _ in models.items()
            ]
        ]
    return Application(urls, debug=True)

if __name__ == '__main__':
    app = make_app()
    app.listen(8080, address='0.0.0.0')
    IOLoop.instance().start()
