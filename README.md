# STSM
The Pytorch implemention for the paper "Predicting high-dimensional time series data with spatial, temporal and global information"

run code like 

```python
`python run.py -dataset Lorentz -model STSM -lr 0.003 -cl False -epoch 100 -bs 16 -dr 0.5 -tpt 40 -ebl 19 -ttl 100000 -si 159 -ei 100000 -itv 59 -tv 0 --tunits 256 30 --sunits 90 10 --fcunits 512 512 256 128 --kw 1 2 3 4 --kn 30 30 30 30 -mm 1`
```
