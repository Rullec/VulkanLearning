from nvidia.dali.pipeline import Pipeline
import nvidia.dali.fn as fn

def DALIDataAugPipeline(batch_size, num_threads, device_id, external_data):
    pipe = Pipeline(batch_size=batch_size * external_data.num_of_view,
                    num_threads=num_threads,
                    device_id=device_id)
    with pipe:
        imgs, labels = fn.external_source(source=external_data, num_outputs=2)
        # st = time.time()
        # rand = fn.random.uniform(range=(-50, 50), shape=2)
        # mt = fn.transforms.translation(offset=rand)
        # imgs = fn.warp_affine(imgs, matrix=mt, inverse_map=False)

        imgs = imgs.gpu()
        # st = time.time()
        angle = fn.random.uniform(range=(-10.0, 10.0))
        imgs = fn.rotate(imgs, angle=angle, keep_size=True)
        imgs = fn.noise.gaussian(
            imgs,
            mean=0.0,
            stddev=0.04,
        )
        imgs = fn.gaussian_blur(imgs, sigma=0.1, window_size=5)
        # ed = time.time()
        # print(f"aug cost {ed - st}")
        pipe.set_outputs(imgs, labels)
    return pipe
