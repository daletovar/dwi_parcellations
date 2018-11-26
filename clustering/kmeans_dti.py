import pandas as pd
import numpy as np 
import os
import joblib
import shutil
import nibabel as nib
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Enter subjects
subs = ['sub-A00028266','sub-A00028287','sub-A00028339','sub-A00028340','sub-A00028352']

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
        indir = studydir + '/' + subID + '/' + session + '/dwi/processed.bedpostX/MPFC'
        outdir = indir + '/clusters' 
        sessiondir = studydir + '/' + subID + '/' + session 
        if os.path.isdir(sessiondir) == False:
            print("No %s directory for subject %s" % (session, subID))
        else:
            if os.path.isdir(outdir) == False:
                os.makedirs(outdir)

        os.chdir(indir)

        # make text files
        os.system('cat fdt_matrix2.dot > matrix2.txt')
        os.system('cat coords_for_fdt_matrix2 > seed_coords.txt') 

        #read in file 
        file = open('matrix2.txt')
        data = file.readlines()
        seed = []
        mask = []
        streams =[]
        for line in data:
                split = line.split('  ')
                seed.append(int(split[0]))
                mask.append(int(split[1]))
                streams.append(float(split[2].replace('/n', '')))
        df = pd.DataFrame()
        df['seed_voxels'] = pd.Series(seed)
        df['mask_voxels'] = pd.Series(mask)
        df['stream_line_count'] = pd.Series(streams)

        print('transforming into sparse matrix')
        # transform into sparse matrix
        rows = set([mask for mask in range(1,df['mask_voxels'].max() + 1)])  
        cols = set([seed for seed in range(1,df['seed_voxels'].max() + 1)])
        mat = pd.DataFrame(index = rows, columns = cols )
        mat = mat.fillna(0)

        # fill in dataframe
        print('filling matrix')
        i = 0
        for value in df['stream_line_count']:
            mat[df.iloc[i,0]][df.iloc[i,1]] = value   
            i += 1
            #print(i)

        # save matrix (just in case)
        joblib.dump(mat, 'clusters/sparse_matrix2.pkl')


        # load very large matrix (if need be)
        #mat = joblib.load('../MPFC_matrix.pkl')


        # transpose and calculate cross-correlation
        mat = mat.T
        CC = 1 + np.corrcoef(mat)
        CC = np.nan_to_num(CC)

        print('generating clusters for session %s subject %s' % (session, subID))

        # kmeans
        kmeans2 = KMeans(n_clusters=2)
        kmeans3 = KMeans(n_clusters=3)
        kmeans4 = KMeans(n_clusters=4)
        kmeans5 = KMeans(n_clusters=5)
        kmeans6 = KMeans(n_clusters=6)

        # fit model
        kmeans2.fit(CC)
        kmeans3.fit(CC)
        kmeans4.fit(CC)
        kmeans5.fit(CC)
        kmeans6.fit(CC)
        # make clusters
        clusters2 = kmeans2.predict(CC)
        clusters3 = kmeans3.predict(CC)
        clusters4 = kmeans4.predict(CC)
        clusters5 = kmeans5.predict(CC)
        clusters6 = kmeans6.predict(CC)
        # get coords data
        file = open('seed_coords.txt')
        data = file.readlines()
        x,y,z = [],[],[]
        voxel_num = [] 
        for line in data:
                split = line.split('  ')
                x.append(int(split[0]))
                y.append(int(split[1]))
                z.append(int(split[2].replace('/n', '')))
                voxel_num.append(int(split[4].replace('/n', '')))
        seed_df = pd.DataFrame()
        seed_df['seed_x'] = pd.Series(x)
        seed_df['seed_y'] = pd.Series(y)
        seed_df['seed_z'] = pd.Series(z)
        seed_df['voxel_num'] = pd.Series(voxel_num)

        # add cluster data
        seed_df['cluster2'] = pd.Series(clusters2) + 1
        seed_df['cluster3'] = pd.Series(clusters3) + 1
        seed_df['cluster4'] = pd.Series(clusters4) + 1
        seed_df['cluster5'] = pd.Series(clusters5) + 1
        seed_df['cluster6'] = pd.Series(clusters6) + 1
        
        
        #### save to a file
        
        print('saving results for session %s subject %s' % (session, subID))
        # specify dimensions 
        dims = (91,109,91)
        k2 = np.zeros(dims)
        k3 = np.zeros(dims)
        k4 = np.zeros(dims)
        k5 = np.zeros(dims)
        k6 = np.zeros(dims)
        # header information
        MNI = nib.load('/usr/local/fsl/data/standard/MNI152_T1_2mm.nii.gz')
        header = MNI.header
        header.set_data_dtype(k2.dtype)  
        # add clusters to file
        for i in range(seed_df.shape[0]):
            k2[seed_df['seed_x'][i]][seed_df['seed_y'][i]][seed_df['seed_z'][i]] = seed_df['cluster2'][i]
            k3[seed_df['seed_x'][i]][seed_df['seed_y'][i]][seed_df['seed_z'][i]] = seed_df['cluster3'][i]
            k4[seed_df['seed_x'][i]][seed_df['seed_y'][i]][seed_df['seed_z'][i]] = seed_df['cluster4'][i]
            k5[seed_df['seed_x'][i]][seed_df['seed_y'][i]][seed_df['seed_z'][i]] = seed_df['cluster5'][i]
            k6[seed_df['seed_x'][i]][seed_df['seed_y'][i]][seed_df['seed_z'][i]] = seed_df['cluster6'][i]
        images = [k2,k3,k4,k5,k6]
        names = ['k2','k3', 'k4', 'k5', 'k6']
        i = 0
        for image in images:
            header['cal_max'] = image.max()
            header['cal_min'] = image.min()
            img = nib.nifti1.Nifti1Image(image, None, header=header)
            filename = 'clusters/MPFC_%s.nii.gz' % names[i] 
            img.to_filename(filename)    
            i += 1
        print('finished session %s subject %s' % (session, subID))