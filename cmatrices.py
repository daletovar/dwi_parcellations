from scipy.sparse import coo_matrix
from scipy.stats import spearmanr
from copy import deepcopy
import joblib
import nibabel as nib 
import numpy as np
import pandas as pd  


def mask_img(img,val):
    """masks an image at a given value - borrowed from de la Vega"""
    img = deepcopy(img)
    data = img.get_data()
    data[:] = np.round(data)
    data[data!=val] = 0
    data[data==val] = 1
    return img

class NeedsNewName(object):
    """ The core object for working with sparse connectivity matrices.
    
    Args:
       data (scipy coo_matrix): The connectivity matrix, whole-brain-by-whole-brain.
       reference (nibabel nifti1): This nifti file holds the identities of the voxels
           in the connectivity matrix. FDT matrices are labeled 'voxel 1, 2, etc. This nifti 
           is the conversion between FDT coordinates and nibabel coordinates.
       mask (nibabel nifti1): A mask for the columns of the connectivity matrix. Most
           often will be a gray-matter mask."""
    
    

    def __init__(self,data,reference,mask=None):

        self.data = joblib.load(data)
        self.reference = reference.get_data()
        self.mask = mask
        
    def init_nifti(self,img_data):
        """for saving images """
        header = self.reference.header
        header.set_data_dtype(img_data.dtype) 
        header['cal_max'] = img_data.max()
        header['cal_min'] = img_data.min()
        return nib.nifti1.Nifti1Image(img_data, affine=self.reference.affine,header=header)
    
    def get_roi_matrix(self,img):
        """a function for generating a connectivity matrix between every voxel in 
        an roi and every voxel in the gray-matter mask"""
        
        r,c,vals = self.data.row, self.data.col, self.data.data
        r,c = r.astype(int), c.astype(int)

        #constructing roi_matrix
        img_data = img.get_data()
        roi_coords = self.reference[np.where(img_data==1)]

        roi_matrix_indices = []
        for i in roi_coords:
            roi_matrix_indices = np.concatenate((roi_matrix_indices,np.where(r==i)), axis=None)
        
        roi_matrix_indices = roi_matrix_indices.astype(int)
        roi_rows = r[roi_matrix_indices].astype(int)
        roi_cols = c[roi_matrix_indices].astype(int)
        roi_vals = vals[roi_matrix_indices]

        # this is for renaming the rows so that the matrix has the proper shape
        reduced_rows = roi_rows
        for i,vox in enumerate(vals,0):
            reduced_rows[np.where(reduced_rows==vox)] = i+1

        return coo_matrix((roi_vals,(reduced_rows,roi_cols)))
    
    def cluster(self,img,model,n_clusters=3, method='pearson'):
        """runs a clustering algorithm on a given roi"""
        
        roi_coords = self.reference[np.where(img.get_data()==1)]
        mat  = self.get_roi_matrix(img)
        mat = mat.toarray()
        # matrices generated with scipy coo_matrix have a 0 row and column, we'll remove them
        mat = np.delete(mat,0,axis=0) 
        mat = np.delete(mat,0,axis=1)
        
        if method=='pearson':
            CC = np.corrcoef(mat)
        elif method=='spearman':
            CC = spearmanr(mat)[0]
        else:
            raise Exception('method should be either pearson or spearman. \
                            The method was: {}'.format(method))
        CC = np.nan_to_num(mat)
        labels = model(n_clusters=n_clusters).fit_predict(CC) + 1
        clusters = np.zeros([91,109,91])        
        # this is a new way that I'm trying to convert clustering results to a nifti
        for i in range(1,labels.max() + 1):
            indices = np.where(labels==i)
            indices = np.array(indices)
            indices = indices.reshape(indices.shape[1])
            cluster_indices = roi_coords[indices].astype(int)
            clusters[np.where(np.isin(self.reference,cluster_indices))] = i
        
        return self.init_nifti(clusters)


    def sum_streamline_count(self,img):
        """generates an array of the streamline count between a cluster and each voxel in the rest of the brain """
        
        mat = self.get_roi_matrix(img)
        vals = mat.toarray().sum(axis=1)   # get the total streamline count by summing the values of each column
        vals = np.delete(vals,0) # the zero voxel doesn't exist so we'll remove it
        return vals
            
    def get_cluster_similarity(self,img,method='pearson'):
        """for comparing the similarity of connectivity distributions bewteen different clusters.
        returns a correlation matrix """
        
        connectivity_vectors = []
        for i in range(1,img.get_data().max() + 1):
            cluster = mask_img(img,i)
            connectivity_vectors[i-1] = self.sum_streamline_count(cluster)
        mat = np.vstack((connectivity_vectors[:]))
        if method=='pearson':
            CC = np.corrcoef(mat)
        elif method=='spearman':
            CC = spearmanr(mat)[0]
        else:
            raise Exception('method should be either pearson or spearman. \
                            The method was: {}'.format(method))
        return np.nan_to_num(CC)

    def get_paths(self,img):  # still working on it
        """creates fsl fdt_paths-style images for each cluster in an image"""
        
        images = [np.zeros([91,109,91]) for i in range(img.get_data().max())]
        
        for i in range(img.get_data().max()):
            cluster = mask_img(img,i+1)
            connections = self.sum_streamline_count(cluster)
            for j in range(len(connections)):
                images[i][np.where(self.reference==j+1)] = connections[j]
        
        return [self.init_nifti(image) for image in images]
                
    
    def connections_to_targets(self,img,targets,labels=None):
        """given an roi it returns a dataframe of connections between the roi and each of the targets """

        stat_map = self.get_paths(img)
        stat_data = stat_map.get_data()
        target_data = nib.load(targets).get_data().round()
        connections = [stat_data[np.where(target_data==i).sum()] for i in range(1,target_data.max() + 1)]
        df = pd.DataFrame()
        df['connections'] = pd.Series(connections)
        
        if labels is not None:
            df['labels'] = pd.Series(labels)
        
        return df


    # work in progress
    #def find_the_biggest(self,*args):
    #    """hard segmentation based on fdt find_the_biggest """ 
    #    labels = np.arange(1,len(args)+1)
    #    brain_data = [nib.load(arg).get_data() for arg in args]
    #    segmented_brain = np.zeros([91,109,91])
    #    for_completement = np.arange(0,len(args))
    #    for i in range(len(brain_data)):
    #        complement = np.delete(for_completement,i)
    #        indices = np.where(brain_data[i]>brain_data[j] for j in complement) # not real python syntax
    #        segmented_brain[indices] = labels[i]
    #    return self.init_nifti(segmented_brain)
    
    # work in progress
    #def network(self,regions,labels=None):
    #    """returns a dataframe of streamline counts between all regions"""
    #    region_data = nib.load(regions).get_data()
    #    
    #    # if no labels are provided, regions will have numeric labels
    #    if labels is None:
    #        labels = np.arange(region_data.max())
    #    
    #    df = pd.DataFrame(columns=labels,index=labels)
    #    
    #    for i in range(1,region_data.max()+1):
    #        region = mask_img(regions)
    #        region_connections = self.connections_to_targets(region,regions)
    #        
    #        
    #    return df

    def save(self,filename):
        joblib.dump(self,filename)

    def load(self,filename):
        joblib.load(filename)

    
                





            
    
