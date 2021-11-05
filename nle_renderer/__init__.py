from .renderer import AsyncRenderer

class Renderer(AsyncRenderer):
    def __init__(self):
        super().__init__()
    
    def set_network(self, pkl):
        if not self._cur_args \
            or 'pkl' not in self._cur_args \
            or pkl is not self._cur_args['pkl']:
            self.set_args(pkl=pkl)

    def set_latents(self, latents):
        self.set_args(latents=latents)

    def set_seed(self, seeds):
        self.set_args(w0_seeds=seeds)

    def get(self):
        return self.get_result()
