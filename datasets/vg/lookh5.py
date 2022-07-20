import h5py
with h5py.File('VG-SGG.h5',"r") as f:
    for key in f.keys():
        print(f[key], key, f[key].name)
