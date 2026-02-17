from rfdetr import RFDETRBase

model = RFDETRBase()

model.train(
    dataset_dir=data_path,
    epochs=10,
    batch_size=4,
    grad_accum_steps=4,
    lr=1e-4,
    output_dir=out_path
)