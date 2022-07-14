from colossalai.amp import AMP_TYPE

# fp16 = dict(
#     mode=AMP_TYPE.TORCH
# )

parallel = dict(
     data=2,
     # pipeline=1,
     tensor=dict(size=2, mode='1d')
) 

inspect_time = False