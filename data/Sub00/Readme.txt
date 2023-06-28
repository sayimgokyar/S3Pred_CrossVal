This file contains the definitons of the data for a fictitious participants called Sub00. All the othe subjec folders uses the same convention for naming and data definitions.

  1- Sub00Run01_iteration001.B1P: This is the B1+ map provided by the scanner for the Sub00, Run01, scan 001. Most of the participants were scanned with different pTx settings (iteration) in one session (Run01), but some participants may be scanned multiple times at different days (Run02, Run03 etc). Scans from different days registered by using SPM12, and RF simulations repeated with a translated head model of the same subject. Then, SAR maps generated.

  2- Sub00Run01_iteration001.B1M: This is the phase-reversed B1+ map calculated by using the channel specific B1+ maps multiplied with the phase-reversed channel weights of the scan. Those weights are already reported as a separate .xlsx file.
  
  3- Sub00Run01_iteration001.T1: This is the MRI data of the given scan. In each MRI session, a participant may be scanned multiple times (30-50 times). For each scan, we have a new MRI data cube.

  4- Sub00Run01_iteration001.SAR: This is the ground truth data for the given RF shim settings. This data is calculated by using Sim4Life (by zmt.swiss) software and generic pTx RF coils reported in the original manuscript.

All of the above data is saved by using the following matlab commands. File extensions changed in the filename variable, and all of the process repeated for the number of scans for each scan session.

imageresolution = [128 128 128];    %Output data resolution (Default 3D:128)
Data_ROI: This is the interpolated data with the above dimensions. interpolation coordinates are identical for a given Run (i.e., assumes subject did not move during a 1hr long scanning session).
ii: index of the scan,

  filename=['Sub00Run01_iteration', num2str(ii,'%03d'), '.B1P'];
  if isfile(filename)
      delete(filename)
  end
h5create(filename, '/ds', imageresolution,'Datatype','single'); h5write(filename, '/ds', Data_ROI);

The above data can be read by using the following python comand in the generator files:

with h5py.File(filename,'r') as f:
    data = f['ds'][:] # this should return a 3D array with 128x128x128 in size.

