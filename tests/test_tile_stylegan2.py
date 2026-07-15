import unittest
from unittest import mock

import numpy as np
import torch

from tile_stylegan2 import TiledStyleGAN2Renderer, coordinate_noise, lcs_noise
from tile_stylegan2_video import quadratic_ease_in_out, slerp_ws
from training import networks_stylegan2


def make_generator(resolution=32):
    torch.manual_seed(7)
    G = networks_stylegan2.Generator(
        z_dim=8,
        c_dim=0,
        w_dim=8,
        img_resolution=resolution,
        img_channels=3,
        channel_base=128,
        channel_max=32,
        num_fp16_res=0,
        use_noise=True,
        mapping_kwargs=dict(num_layers=2),
    ).eval()
    with torch.no_grad():
        for module in G.modules():
            if isinstance(module, networks_stylegan2.SynthesisLayer) and module.use_noise:
                module.noise_strength.fill_(0.25)
    return G


class TiledStyleGAN2Tests(unittest.TestCase):
    def test_split_tail_matches_normal_synthesis(self):
        G = make_generator()
        renderer = TiledStyleGAN2Renderer(G)
        ws = renderer.map_seeds([11])

        with torch.inference_mode():
            expected = G.synthesis(ws, noise_mode="none", force_fp32=True, fused_modconv=False)
            x, img = renderer.run_prefix(ws, split_resolution=8)
            actual = renderer.run_tail(x, img, ws, split_resolution=8)

        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    def test_arbitrary_feature_canvas_shape(self):
        G = make_generator()
        renderer = TiledStyleGAN2Renderer(G)
        canvas = renderer.build_feature_canvas(
            rows=2,
            cols=3,
            split_resolution=8,
            core_size=4,
            seed=20,
            batch_size=3,
        )
        tail_ws = renderer.map_seeds([999])
        output = renderer.render_tail_chunked(canvas, tail_ws, chunk_size=5, halo=8, noise_seed=123)
        self.assertEqual(tuple(output.shape), (1, 3, 32, 48))

    def test_chunked_tail_matches_whole_canvas(self):
        G = make_generator()
        renderer = TiledStyleGAN2Renderer(G)
        canvas = renderer.build_feature_canvas(
            rows=2,
            cols=3,
            split_resolution=8,
            core_size=4,
            seed=30,
            batch_size=2,
        )
        tail_ws = renderer.map_seeds([1001])

        with torch.inference_mode():
            expected = renderer.run_tail(
                canvas.x,
                canvas.img,
                tail_ws,
                split_resolution=8,
                noise_seed=456,
            )
            actual = renderer.render_tail_chunked(
                canvas,
                tail_ws,
                chunk_size=5,
                halo=8,
                noise_seed=456,
            )

        torch.testing.assert_close(actual, expected.cpu(), rtol=1e-5, atol=2e-5)

    def test_coordinate_noise_is_stable_across_overlapping_regions(self):
        whole = coordinate_noise(12, 15, 0, 0, seed=99, stream=32)
        region = coordinate_noise(5, 6, 3, 4, seed=99, stream=32)
        torch.testing.assert_close(region, whole[:, :, 3:8, 4:10], rtol=0, atol=0)

    def test_tail_noise_seed_changes_output(self):
        G = make_generator()
        renderer = TiledStyleGAN2Renderer(G)
        canvas = renderer.build_feature_canvas(
            rows=1,
            cols=2,
            split_resolution=8,
            core_size=4,
            seed=40,
            batch_size=2,
        )
        tail_ws = renderer.map_seeds([1002])
        first = renderer.run_tail(canvas.x, canvas.img, tail_ws, 8, noise_seed=1)
        second = renderer.run_tail(canvas.x, canvas.img, tail_ws, 8, noise_seed=2)
        self.assertGreater(float((first - second).abs().max()), 1e-4)

    def test_candidate_selection_and_prefix_rgb_control(self):
        G = make_generator()
        renderer = TiledStyleGAN2Renderer(G)
        dropped = renderer.build_feature_canvas(
            rows=2,
            cols=2,
            split_resolution=8,
            core_size=4,
            seed=50,
            batch_size=3,
            candidates_per_tile=3,
            match_width=2,
        )
        self.assertIsNone(dropped.img)
        self.assertEqual(len(dropped.seeds), 4)
        for tile_idx, selected_seed in enumerate(dropped.seeds):
            self.assertIn(selected_seed, range(50 + tile_idx * 3, 53 + tile_idx * 3))

        kept = renderer.build_feature_canvas(
            rows=1,
            cols=1,
            split_resolution=8,
            core_size=4,
            seed=60,
            keep_prefix_rgb=True,
        )
        self.assertIsNotNone(kept.img)

    def test_feature_blend_preserves_shape_and_feathers_seam(self):
        G = make_generator()
        renderer = TiledStyleGAN2Renderer(G)
        common = dict(
            rows=1,
            cols=2,
            split_resolution=8,
            core_size=4,
            seed=65,
            candidates_per_tile=1,
            prefix_noise_mode="none",
        )
        hard = renderer.build_feature_canvas(**common, blend_width=0)
        blended = renderer.build_feature_canvas(**common, blend_width=1)

        self.assertEqual(blended.shape, hard.shape)
        torch.testing.assert_close(blended.x[:, :, :, :3], hard.x[:, :, :, :3])
        torch.testing.assert_close(blended.x[:, :, :, 5:], hard.x[:, :, :, 5:])
        self.assertGreater(
            float((blended.x[:, :, :, 3:5] - hard.x[:, :, :, 3:5]).abs().max()),
            1e-6,
        )

    def test_interpolated_candidate_ws_and_locked_selection(self):
        G = make_generator()
        renderer = TiledStyleGAN2Renderer(G)
        ws = renderer.map_seeds([100, 101, 102, 103]).cpu().reshape(
            2, 2, G.mapping.num_ws, G.w_dim
        )
        with mock.patch.object(
            renderer, "map_seeds", side_effect=AssertionError("seeds should not be mapped")
        ):
            canvas = renderer.build_feature_canvas(
                rows=1,
                cols=2,
                split_resolution=8,
                core_size=4,
                blend_width=1,
                seed=0,
                candidates_per_tile=2,
                selected_candidate_indices=(1, 0),
                candidate_ws=ws,
                prefix_noise_mode="none",
            )
        self.assertEqual(canvas.candidate_indices, (1, 0))
        self.assertEqual(canvas.shape, (4, 8))

    def test_lcs_style_w_plus_slerp(self):
        first = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
        second = torch.tensor([[[0.0, 1.0], [-1.0, 0.0]]])
        torch.testing.assert_close(slerp_ws(first, second, 0.0), first)
        torch.testing.assert_close(slerp_ws(first, second, 1.0), second)
        midpoint = slerp_ws(first, second, 0.5)
        expected = torch.tensor([[[2**-0.5, 2**-0.5], [-2**-0.5, 2**-0.5]]])
        torch.testing.assert_close(midpoint, expected)
        self.assertEqual(quadratic_ease_in_out(0.25), 0.125)

    def test_lcs_noise_matches_original_formula_and_overlaps(self):
        from opensimplex import OpenSimplex

        generator_resolution = 32
        layer_resolution = 8
        seed = 3
        frame = 7
        actual = lcs_noise(
            5,
            6,
            0,
            0,
            seed,
            frame,
            generator_resolution,
            layer_resolution,
        )

        num_slices = min(512, generator_resolution**2 * 16 // layer_resolution**2)
        radius = min(1, max(0.1, 64 / layer_resolution))
        z = np.sin(frame / num_slices * np.pi * 2) * radius
        w = np.cos(frame / num_slices * np.pi * 2) * radius
        simplex = OpenSimplex(seed)
        expected = torch.empty([1, 1, 5, 6])
        for row in range(5):
            for col in range(6):
                expected[0, 0, row, col] = simplex.noise4(row, col, z, w)
        expected = expected.to(torch.bfloat16).to(torch.float32)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

        region = lcs_noise(
            2,
            3,
            2,
            1,
            seed,
            frame,
            generator_resolution,
            layer_resolution,
        )
        torch.testing.assert_close(region, actual[:, :, 2:4, 1:4], rtol=0, atol=0)

        negative = lcs_noise(
            4,
            5,
            -2,
            -1,
            seed,
            frame,
            generator_resolution,
            layer_resolution,
        )
        center = lcs_noise(
            2,
            3,
            0,
            0,
            seed,
            frame,
            generator_resolution,
            layer_resolution,
        )
        torch.testing.assert_close(center, negative[:, :, 2:4, 1:4], rtol=0, atol=0)

    def test_lcs_prefix_uses_global_tile_coordinates(self):
        G = make_generator()
        renderer = TiledStyleGAN2Renderer(G)
        calls = []

        def fake_lcs_noise(
            height,
            width,
            origin_y,
            origin_x,
            seed,
            frame,
            generator_resolution,
            layer_resolution,
        ):
            calls.append((origin_y, origin_x, seed, layer_resolution))
            return torch.zeros([1, 1, height, width])

        with mock.patch("tile_stylegan2.lcs_noise", side_effect=fake_lcs_noise):
            renderer.build_feature_canvas(
                rows=1,
                cols=2,
                split_resolution=8,
                core_size=4,
                seed=80,
                candidates_per_tile=1,
                prefix_noise_mode="lcs",
                noise_seed=10,
            )

        self.assertEqual(
            calls,
            [
                (-2, -2, 10, 8),
                (-2, -2, 11, 8),
                (-2, 2, 10, 8),
                (-2, 2, 11, 8),
            ],
        )

    def test_lcs_noise_chunked_tail_matches_whole_canvas(self):
        G = make_generator()
        renderer = TiledStyleGAN2Renderer(G)
        canvas = renderer.build_feature_canvas(
            rows=2,
            cols=3,
            split_resolution=8,
            core_size=4,
            seed=70,
            batch_size=3,
        )
        tail_ws = renderer.map_seeds([1003])
        expected = renderer.run_tail(
            canvas.x,
            canvas.img,
            tail_ws,
            split_resolution=8,
            noise_seed=1,
            noise_mode="lcs",
            noise_frame=5,
            noise_level=0.75,
        )
        actual = renderer.render_tail_chunked(
            canvas,
            tail_ws,
            chunk_size=5,
            halo=8,
            noise_seed=1,
            noise_mode="lcs",
            noise_frame=5,
            noise_level=0.75,
        )
        torch.testing.assert_close(actual, expected.cpu(), rtol=1e-5, atol=2e-5)


if __name__ == "__main__":
    unittest.main()
