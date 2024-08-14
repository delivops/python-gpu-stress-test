## Check GPU utilization from inside a pod:

```bash
watch -d -n 0.5 nvidia-smi
```

Using GPU requires 3.5GB of video memory per pod (can be managed inside an app by pytorch libary).
Tested GPU has 24GB of video memory which allows to run 6 pods at the same time.

```bash
python main.py --device cuda
```

Logs, no memory limit:

```csv
Pod Name,Last Line
gpu-test-deployment-76c4fbf9c8-4wbxp,Loop count: 2008, Total elapsed time: 493.10 seconds
gpu-test-deployment-76c4fbf9c8-4zh6m,Loop count: 1903, Total elapsed time: 487.76 seconds
gpu-test-deployment-76c4fbf9c8-78sb2,Loop count: 1871, Total elapsed time: 487.75 seconds
gpu-test-deployment-76c4fbf9c8-gcrln,Loop count: 1836, Total elapsed time: 482.63 seconds
gpu-test-deployment-76c4fbf9c8-gtgkb,Loop count: 1810, Total elapsed time: 477.91 seconds
gpu-test-deployment-76c4fbf9c8-h7nkj,Loop count: 1805, Total elapsed time: 477.46 seconds
gpu-test-deployment-76c4fbf9c8-pdf52,
gpu-test-deployment-76c4fbf9c8-vptv4,
gpu-test-deployment-76c4fbf9c8-z2rw5,
gpu-test-deployment-76c4fbf9c8-zndzw,
```

```bash
python main.py --device cpu
```

logs CPU, no memory limit:

```csv
Pod Name,Last Line
gpu-test-deployment-76c4fbf9c8-cft5z,Loop count: 28, Total elapsed time: 496.83 seconds
gpu-test-deployment-76c4fbf9c8-fzfj6,Loop count: 28, Total elapsed time: 490.31 seconds
gpu-test-deployment-76c4fbf9c8-gqvp5,Loop count: 28, Total elapsed time: 492.61 seconds
gpu-test-deployment-76c4fbf9c8-nfbwm,Loop count: 28, Total elapsed time: 481.99 seconds
gpu-test-deployment-76c4fbf9c8-r4s4x,Loop count: 28, Total elapsed time: 489.71 seconds
gpu-test-deployment-76c4fbf9c8-rq7bg,Loop count: 28, Total elapsed time: 483.70 seconds
gpu-test-deployment-76c4fbf9c8-s9htq,Loop count: 28, Total elapsed time: 471.53 seconds
gpu-test-deployment-76c4fbf9c8-sgzt6,Loop count: 29, Total elapsed time: 482.41 seconds
gpu-test-deployment-76c4fbf9c8-t668m,Loop count: 28, Total elapsed time: 465.43 seconds
gpu-test-deployment-76c4fbf9c8-xsp66,Loop count: 28, Total elapsed time: 462.89 seconds
```
