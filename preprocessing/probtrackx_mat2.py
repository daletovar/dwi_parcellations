#probtrackx
import shutil
import os
import numpy

# Enter subject IDs
subs = ['sub-A00028822','sub-A00028842']

# Enter NKI Rockland wave 
nki_session = ['ses-DS2', 'ses-DSA', 'ses-CLGA','ses-NFB2R',
               'ses-NFB3', 'ses-CLG2R', 'ses-CLG4', 'ses-NFB2', 
               'ses-CLG2', 'ses-NFBA']

studydir = '/Users/chavezlabadmin/Documents/rockland_data'
seed = '/Users/chavezlabadmin/Documents/kmeans/reduced_MPFC_mask.nii'
mask = '/Users/chavezlabadmin/Documents/kmeans/reduced_grey_matter_mask.nii.gz'


for session in nki_session:
    
    for subID in subs:
        print('Running %s' % (subID))
        bedpostx_dir = studydir + '/' + subID + '/' + session + '/dwi/processed.bedpostX'
        warp = studydir + '/' + subID + '/' + session + '/anat/processed/FA_2_MNI2mm_fnirt_warp.nii.gz'
        rev_warp = studydir +  '/' + subID + '/' + session + '/anat/processed/FA_2_MNI2mm_inverse_warp.nii.gz'
        MPFC_dir = bedpostx_dir + '/MPFC'
        
        sessiondir = studydir + '/' + subID + '/' + session 
        if os.path.isdir(sessiondir) == False:
            print("No %s directory for subject %s" % (session, subID))
        else:

            command = '/usr/local/fsl/bin/probtrackx2 -x ' + seed + ' -l --onewaycondition --omatrix2 --target2='+ mask + ' ' + ' \
            --pd -c 0.2 -S 2000 --steplength=0.5 -P 5000 --fibthresh=0.01 --distthresh=0.0 --sampvox=0.0 \
            --xfm=' + rev_warp + ' --invxfm=' + warp + ' --forcedir --opd -s ' + bedpostx_dir + '/./merged \
            -m ' + bedpostx_dir + '/./nodif_brain_mask --dir=' + MPFC_dir
        
            os.system(command)
        
        
