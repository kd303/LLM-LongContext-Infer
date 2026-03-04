from nvidia.dali import pipeline_def
import nvidia.dali.fn as fn
import nvidia.dali.types as types

## Number of threads is for typically decoders which are CPU based. A100 has JPEG/WebP decoders.

@autoserialize
@dali.pipeline_def(batch_size=32, num_threads=4, device_id=0)
def clip_preprocessing_pipeline(device="gpu"):
    # 1. External Source: We feed raw JPEG bytes and pre-tokenized IDs
     
    jpegs = fn.external_source(device="cpu", name="IMAGE_INPUT")
   
    # 2. Image Branch (GPU accelerated), mixed actuall falls back to CPU for non JPEG images, for JPEG it uses hardware decoders
    images = fn.decoders.image(jpegs, device="mixed", output_type=types.RGB)
    
    # CLIP standard: Resize to 224x224 and Center Crop
    images = fn.resize(images, resize_shorter=224, interp_type=types.INTERP_CUBIC)
    # Normalization (CLIP Values: mean=[0.481, 0.457, 0.408], std=[0.268, 0.261, 0.275])
    # Note: DALI expect values scaled 0-255 or 0-1. 
    # Here we normalize directly to the CLIP distribution.
    # CHW - Channel Height Width
    pixel_values = fn.crop_mirror_normalize(
        images, crop=(224, 224),
        dtype=types.FLOAT,
        output_layout="CHW",
        mean=[0.48144531 * 255, 0.4578275 * 255, 0.40821073 * 255],
        std=[0.26862954 * 255, 0.26130258 * 255, 0.27577711 * 255]
    )

    return pixel_values