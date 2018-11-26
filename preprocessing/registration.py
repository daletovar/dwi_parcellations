import os
import shutil
import numpy as np

# Enter subject IDs

subs = ['sub-A00008326']
    
# Enter study root directory.
studydir = '/Users/chavezlabadmin/Documents/rockland_data'

# Enter NKI Rockland wave 
nki_session = ['ses-DS2', 'ses-DSA', 'ses-CLGA','ses-NFB2R',
               'ses-NFB3', 'ses-CLG2R', 'ses-CLG4', 'ses-NFB2', 
               'ses-CLG2', 'ses-NFBA']

for session in nki_session:
    
    for subID in subs:

        print('Running %s' % (subID))
        # Assign input and output directories
        indir = studydir + '/' + subID + '/' + session + '/anat'
        outdir = studydir + '/' + subID + '/' + session + '/anat' + '/processed' 
        sessiondir = studydir + '/' + subID + '/' + session 
        if os.path.isdir(sessiondir) == False:
            print("No %s directory for subject %s" % (session, subID))
        else:
            if os.path.isdir(outdir) == False:
                os.makedirs(outdir)

            # Copy anat and FA files for processing
            shutil.copy(indir + '/' + subID + '_' + session + '_T1w.nii.gz', outdir + '/raw_anat.nii.gz')
            shutil.copy(studydir + '/' + subID + '/' + session + '/dwi/processed/rdti_FA.nii.gz', outdir + '/FA.nii.gz')
            os.chdir(outdir)
            
            # run ssroi
            os.system('standard_space_roi raw_anat.nii.gz anat_ssroi.nii.gz -b')
            
            #run brain extraction on ssroi
            os.system('bet2 anat_ssroi.nii.gz anat_brain.nii.gz -f .25 -m')
            
            #linear registration
            os.system('flirt -ref /usr/local/fsl/data/standard/MNI152_T1_2mm_brain \
            -in anat_brain.nii.gz \
            -out anat_2_MNI_flirt.nii.gz \
            -omat anat_2_MNI_flirt.mat \
            -bins 256 -cost corratio -searchrx -90 90 -searchry -90 90 -searchrz -90 90 -dof 6 -interp trilinear')
            
            # inverse matrix
            os.system('convert_xfm -omat anat_flirt_invert -inverse anat_2_MNI_flirt.mat')
            
            #non-linear registration
            os.system('fnirt --in=raw_anat.nii.gz --aff=anat_2_MNI_flirt.mat --cout=anat_2_MNI_fnirt_warp --iout=anat_2_MNI_fnirt --jout=anat_2_MNI_fnirt_jac \
            --config=T1_2_MNI152_2mm --ref=/usr/local/fsl/data/standard/MNI152_T1_2mm.nii.gz \
            --refmask=/usr/local/fsl/data/standard/MNI152_T1_2mm_brain_mask_dil.nii.gz --warpres=10,10,10')
            
            #inverse warp
            os.system('invwarp -w anat_2_MNI_fnirt_warp.nii.gz -o anat_inverse_warp -r raw_anat')
            
            #warp anat 
            os.system('applywarp -i anat_brain.nii.gz -o anat_fnirt \
                      -r /usr/local/fsl/data/standard/MNI152_T1_2mm_brain.nii.gz -w anat_2_MNI_fnirt_warp.nii.gz')
            
            #fit FA to anat_brain with flirt
            os.system('flirt -ref anat_brain.nii.gz \
            -in FA.nii.gz \
            -out FA_2_anat \
            -omat FA_2_anat.mat \
            -bins 256 -cost corratio -searchrx -90 90 -searchry -90 90 -searchrz -90 90 -dof 6 -interp trilinear')
            
            # inverse matrix
            os.system('convert_xfm -omat FA_flirt_invert -inverse FA_2_anat.mat')
            
            #warp FA 
            os.system('applywarp -i FA_2_anat.nii.gz -o FA_fnirt -r /usr/local/fsl/data/standard/MNI152_T1_2mm_brain.nii.gz -w anat_2_MNI_fnirt_warp.nii.gz')
            
            #fit FA to MNI with flirt
            os.system('flirt -ref /Users/chavezlabadmin/Documents/templates/FA_MNI_2mm.nii.gz \
            -in FA.nii.gz \
            -out FA_2_MNI2mm_flirt \
            -omat FA_2_MNI2mm_flirt.mat \
            -bins 256 -cost corratio -searchrx -90 90 -searchry -90 90 -searchrz -90 90 -dof 6 -interp trilinear')
            
            # inverse matrix
            os.system('convert_xfm -omat FA_2_MNI2mm_flirt_invert -inverse FA_2_MNI2mm_flirt.mat')
            
            #non-linear registration
            os.system('fnirt --in=FA.nii.gz --aff=FA_2_MNI2mm_flirt.mat --cout=FA_2_MNI2mm_fnirt_warp --iout=FA_2_MNI2mm_fnirt --jout=FA_2_MNI2mm_fnirt_jac \
            --config=T1_2_MNI152_2mm --ref=/Users/chavezlabadmin/Documents/templates/FA_MNI_2mm.nii.gz \
            --refmask=/Users/chavezlabadmin/Documents/templates/FA_MNI_2mm_mask.nii.gz --warpres=10,10,10')
            
            #inverse warp
            os.system('invwarp -w FA_2_MNI2mm_fnirt_warp -o FA_2_MNI2mm_inverse_warp -r FA')
            
            #warp FA to MNI 
            os.system('applywarp -i FA_2_MNI2mm_flirt.nii.gz -o FA_MNI_fnirt -r /Users/chavezlabadmin/Documents/templates/FA_MNI_2mm.nii.gz -w FA_2_MNI2mm_fnirt_warp.nii.gz')
            
            
            
            
            
            
            
            