Copy the exported deployable actor to this directory as `policy.onnx`.

The model contract is:

- actor observation: 275 values (55 deployable values x five frames);
- action: 16 values (12 leg position actions, then four wheel velocity actions);
- control rate: 50 Hz.
