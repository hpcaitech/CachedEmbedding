from colossalai.amp import AMP_TYPE

# fp16 = dict(
#     mode=AMP_TYPE.TORCH
# )

parallel = dict(
    tensor=dict(
        mode="1d",
        size=2
    )
)

inspect_time = True
