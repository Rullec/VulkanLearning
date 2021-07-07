from nvidia.dali.pipeline import Pipeline
import nvidia.dali.fn as fn


def DALIDataAugPipeline(batch_size, num_threads, device_id, external_data):
    pipe = Pipeline(batch_size=batch_size * external_data.num_of_view,
                    num_threads=num_threads,
                    device_id=device_id)
    with pipe:
        imgs, labels = fn.external_source(source=external_data, num_outputs=2)
        
        imgs = imgs.gpu()
        mt = fn.transforms.rotation(angle=fn.random.uniform(range=(-5.0,
                                                                   5.0)))
        mt = fn.transforms.translation(
            offset=fn.random.uniform(range=(-30, 30), shape=2))

        imgs = fn.warp_affine(imgs, matrix=mt, fill_value=0, inverse_map=False)
        imgs = fn.noise.gaussian(
            imgs,
            mean=0.0,
            stddev=0.04,
        )
        imgs = fn.gaussian_blur(imgs, sigma=0.1, window_size=5)


        pipe.set_outputs(imgs, labels)
    return pipe
