import numpy as np
import h5py
from larcv import larcv
from ROOT import TChain

#input_file = "/vols/sbn/uboone/darkTridents/data/CNN_training/MPID_test_set_full.root"
image_tree = "image2d_image2d_binary_tree"
#output_file = "/vols/sbn/uboone/ll4420/dark_tridents_wspace/hdf5/output_hdf5/MPID_test_set_full.h5"
input_file = "/vols/sbn/uboone/darkTridents/data/larcv_files/run3_data/run3_NuMI_beamon_larcv_cropped_full_set.root"
output_file = "/vols/sbn/uboone/ll4420/dark_tridents_wspace/hdf5/output_hdf5/run3_NuMI_beamon_full.h5"

# Determined based on visual inspection of the images
ADC_THRESHOLD = 5

chain = TChain(image_tree)
chain.AddFile(input_file)
n_events = chain.GetEntries()
print(f"Total events: {n_events}")

with h5py.File(output_file, 'w') as f:
    for i in range(n_events):
        chain.GetEntry(i)
        cpp_obj = chain.image2d_image2d_binary_branch
        image = larcv.as_ndarray(cpp_obj.as_vector()[0])

        run = cpp_obj.run()
        label = 1 if run == 100 else 0

        time_coords, wire_coords = np.where(image > ADC_THRESHOLD)
        adc_values = image[time_coords, wire_coords]
        spacepoints = np.column_stack([wire_coords, time_coords, adc_values])

        grp = f.create_group(f'event_{i}')
        grp.create_dataset('spacepoints', data=spacepoints)
        grp.attrs['run'] = run
        grp.attrs['subrun'] = cpp_obj.subrun()
        grp.attrs['event'] = cpp_obj.event()
        grp.attrs['label'] = label

        if i % 1000 == 0:
            print(f"Processed {i}/{n_events}")

print(f"Saved to {output_file}")