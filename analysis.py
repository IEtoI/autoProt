import os
from importlib import resources
from pathlib import Path
from scipy.stats import ttest_1samp, ttest_ind, zscore
from scipy.spatial import distance
from scipy.cluster import hierarchy
from scipy.cluster.hierarchy import maxdists,fcluster
from statsmodels.stats import multitest as mt
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score,calinski_harabasz_score, davies_bouldin_score
import matplotlib.pylab as plt
import pylab as pl
import colorsys
import seaborn as sns
from operator import itemgetter
import itertools
from Bio import Entrez
import time
from autoprot import visualization as vis
import wordcloud as wc
import warnings
#might want to enable warnings for debugging
#disabled them for copy vs view pandas warnings (pretty annoying)
warnings.filterwarnings('ignore')

#chekc where this is actually used and make it local
cmap = sns.diverging_palette(150, 275, s=80, l=55, n=9)

def ttest(df, reps, cond="", mean=True):
    """
    @params 
    :df: pandas data frame
    :reps: the replicates to be included in the statistical test either 
    a list of the replicates or a list containing two list with the respective replicates
    :cond: the name of the condition. This is used for naming the returned results
    :mean: whether to calculate the logFC of the provided data
    For the one sample ttest log2 transformed ratios are expected
    For the two sample ttest log transformed intensities are expected
    if mean=True for two sample ttest the log2 ratio of the provided replicates will
    be calculated
    """
    cond = '_'+cond
    if isinstance(reps[0], list) and len(reps) == 2:
        df[f"pValue{cond}"] = df[reps[0]+reps[1]].apply(lambda x: np.ma.filled(ttest_ind(x[:len(reps[0])], x[len(reps[0]):], nan_policy="omit")[1],np.nan),1).astype(float)
        df[f"score{cond}"] = -np.log10(df[f"pValue{cond}"])
        if mean == True:
            df[f"logFC{cond}"] = np.log2(pd.DataFrame(df[reps[0]].values / df[reps[1]].values).mean(1)).values
        
    else:
        df[f"pValue{cond}"] = df[reps].apply(lambda x: np.ma.filled(ttest_1samp(x, nan_policy="omit", popmean=0)[1],np.nan),1).astype(float)
        df[f"score{cond}"] = -np.log10(df[f"pValue{cond}"])
        if mean == True:
            df[f"logFC{cond}"] = df[reps].mean(1)
    return df


def adjustP(df, pCol, method="fdr_bh"):
    """
    Wrapper for statsmodels.multitest
    Methods include:
    'b': 'Bonferroni',
    's': 'Sidak',
    'h': 'Holm',
    'hs': 'Holm-Sidak',
    'sh': 'Simes-Hochberg',
    'ho': 'Hommel',
    'fdr_bh': 'FDR Benjamini-Hochberg',
    'fdr_by': 'FDR Benjamini-Yekutieli',
    'fdr_tsbh': 'FDR 2-stage Benjamini-Hochberg',
    'fdr_tsbky': 'FDR 2-stage Benjamini-Krieger-Yekutieli',
    'fdr_gbs': 'FDR adaptive Gavrilov-Benjamini-Sarkar'
    """
    df[f"adj.{pCol}"] = mt.multipletests(df[pCol], method=method)[1]
    return df


class autoPCA:
    """
    The autoPCA class allows conducting a PCA 
    furhter it encompasses a set of helpful visualizations
    for further investigating the results of the PCA
    It needs the matrix on which the PCA is performed
    as well as row labels (rlabels)
    and column labels (clabels) corresponding to the
    provided matrix.
    
    ToDo: Add interactive 3D scatter plot
    """
    def __init__(self, X, rlabels, clabels, batch=None):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        self.X = X.dropna()
        self.label = clabels
        self.rlabel = rlabels
        self.batch = batch
        self.pca, self.forVis =  self._performPCA(X, clabels)
        self.Xt = self.pca.transform(self.X)
        self.expVar = self.pca.explained_variance_ratio_

    @staticmethod
    def _performPCA(X, label):
        """
        performs pca and also generates forVis dataframe
        in this the loadings of the components are stored for visualization
        """
        pca = PCA().fit(X.dropna())
        forVis = pd.DataFrame(pca.components_.T)
        forVis.columns = [f"PC{i}" for i in range(1, min(X.shape[0], X.T.shape[0])+1)]
        forVis["label"] = label
        return pca, forVis


    def scree(self, figsize=(15,5)):
        if not isinstance(self.pca,PCA):
            raise TypeError("This is a function to plot Scree plots. Provide fitted sklearn PCA object.")

        eigVal = self.pca.explained_variance_
        cumVar = np.append(np.array([0]), np.cumsum(self.expVar))

        plt.figure(figsize=figsize)
        plt.subplot(121)
        plt.plot(range(1,len(eigVal)+1), eigVal, marker="o", color="teal",
                markerfacecolor='purple')
        plt.ylabel("Eigenvalues")
        plt.xlabel("# Component")
        plt.title("Scree plot")
        sns.despine()

        plt.subplot(122)
        plt.plot(range(1, len(cumVar)+1), cumVar, ds="steps", color="teal")
        plt.xticks(range(1,len(eigVal)+1))
        plt.ylabel("explained cumulative variance")
        plt.xlabel("# Component")
        plt.title("Explained variance")
        sns.despine()


    def corrComp(self, annot=False):
        plt.figure()
        sns.heatmap(self.forVis.filter(regex="PC"), cmap=sns.color_palette("PuOr", 10), annot=annot)
        yp = [i+0.5 for i in range(len(self.label))]
        plt.yticks(yp, self.forVis["label"], rotation=0);
        plt.title("")


    def barLoad(self, pc=1, n=25):
        """
        Function that plots the loadings of a given component in a barplot
        loadings are sorted
        @parameter
        pc: component to draw given as an integer 
        """
        PC =  f"PC{pc}"
        forVis = self.forVis.copy()
        forVis[f"{PC}_abs"] = abs(forVis[PC])
        forVis["color"] = "negative"
        forVis.loc[forVis[PC]>0, "color"] = "positive"
        forVis = forVis.sort_values(by=f"{PC}_abs", ascending=False)[:n]
        plt.figure()
        ax = plt.subplot()
        sns.barplot(x=forVis[PC], y=forVis["label"], hue=forVis["color"], alpha=.5,
                    hue_order=["negative", "positive"], palette=["teal", "purple"])
        ax.get_legend().remove()
        sns.despine()
        
        
    def returnLoad(self, pc=1, n=25):
        PC =  f"PC{pc}"
        forVis = self.forVis.copy()
        forVis[f"{PC}_abs"] = abs(forVis[PC])
        forVis = forVis.sort_values(by=f"{PC}_abs", ascending=False)[:n]
        return forVis[[PC, "label"]]


    def scorePlot(self, pc1=1, pc2=2, labeling=False, file=None, figsize=(5,5)):
        
        x = self.Xt[::,pc1-1]
        y = self.Xt[::,pc2-1]
        plt.figure(figsize=figsize)
        if self.batch is None:
            forVis = pd.DataFrame({"x":x,"y":y})
            sns.scatterplot(data=forVis, x="x", y="y")
        else:
            forVis = pd.DataFrame({"x":x,"y":y, "batch":self.batch})
            sns.scatterplot(data=forVis, x="x", y="y", hue=forVis["batch"])
        forVis["label"] = self.rlabel
        

        plt.title("Score plot")
        plt.xlabel(f"PC{pc1}\n{round(self.expVar[pc1-1]*100,2)} %")
        plt.ylabel(f"PC{pc2}\n{round(self.expVar[pc2-1]*100,2)} %")
        # plt.scatter(x,y)
        if labeling is True:
            ss = forVis["label"]
            xx = forVis["x"]
            yy = forVis["y"]
            for x,y,s in zip(xx,yy,ss):
                plt.text(x,y,s)
        sns.despine()
        
        if file is not None:
            plt.savefig(fr"{file}/ScorePlot.pdf")


    def loadingPlot(self, pc1=1, pc2=2, labeling=False, figsize=(7,7)):
        plt.figure(figsize=figsize)
        if self.batch is None or len(self.batch) != self.forVis.shape[0]:
            sns.scatterplot(data=self.forVis, x=f"PC{pc1}",
            y=f"PC{pc2}",edgecolor=None)
        else:
            sns.scatterplot(data=self.forVis, x=f"PC{pc1}",
            y=f"PC{pc2}",edgecolor=None, hue=self.batch)
        sns.despine()
        if labeling is True:
            ss = self.forVis["label"]
            xx = self.forVis[f"PC{pc1}"]
            yy = self.forVis[f"PC{pc2}"]
            for x,y,s in zip(xx,yy,ss):
                plt.text(x,y,s)
    
    
    def biPlot(self, pc1=1, pc2=2, numLoad="all", figsize=(3,3), **kwargs):
        
        x = self.Xt[::,pc1-1]
        y = self.Xt[::,pc2-1]
        plt.figure(figsize=figsize)
        plt.scatter(x,y, color="lightgray", alpha=0.5, linewidth=0, **kwargs)

        temp = self.forVis[[f"PC{pc1}", f"PC{pc2}"]]
        temp["label"] = self.label
        temp = temp.sort_values(by=f"PC{pc1}")
        
        if numLoad == "all":
            loadings = temp[[f"PC{pc1}", f"PC{pc2}"]].values
            labels = temp["label"].values
        else:
            loadings = temp[[f"PC{pc1}", f"PC{pc2}"]].iloc[:numLoad].values
            labels = temp["label"].iloc[:numLoad].values
            
        xscale = 1.0 / (self.Xt[::,pc1-1].max() - self.Xt[::,pc1-1].min())
        yscale = 1.0 / (self.Xt[::,pc2-1].max() - self.Xt[::,pc2-1].min())
        xmina = 0
        xmaxa = 0
        ymina = 0
        ymaxa = 0

        for l, lab in zip(loadings, labels):
            #plt.plot([0,l[0]/xscale], (0, l[1]/yscale), color="purple")
            plt.arrow(x=0, y=0,dx=l[0]/xscale, dy= l[1]/yscale, color="purple",
                     head_width=.2)
            plt.text(x=l[0]/xscale, y=l[1]/yscale, s=lab)
            
            if l[0]/xscale < xmina:
                xmina = l[0]/xscale
            elif l[0]/xscale > xmaxa:
                xmaxa = l[0]/xscale
                
            if l[1]/yscale < ymina:
                ymina = l[1]/yscale
            elif l[1]/yscale > ymina:
                ymina = l[1]/yscale

        plt.xlabel(f"PC{pc1}\n{round(self.expVar[pc1-1]*100,2)} %")
        plt.ylabel(f"PC{pc2}\n{round(self.expVar[pc2-1]*100,2)} %")
        sns.despine()

    
    def pairPlot(self, n=0):
        """
        function draws a pair plot of for pca for the given
        number of dimensions (or all if none is specified.
        !Be carful for large data this might crash you PC -> better specify n!
        """
        #must be prettyfied quite a bit....
        if n == 0:
            n = self.Xt.shape[0]
        
        forVis = pd.DataFrame(self.Xt[:,:n])
        i = np.argmin(self.Xt.shape)
        pcs = self.Xt.shape[i]
        forVis.columns = [f"PC {i}" for i in range(1,pcs+1)]
        if self.batch is not None:
            forVis["batch"] = self.batch
            sns.pairplot(forVis, hue="batch")
        else:
            sns.pairplot(forVis)
        
        
    def returnScore(self):
    
        columns = [f"PC{i+1}" for i in range(self.Xt.shape[1])]
        scores = pd.DataFrame(self.Xt, columns=columns)
        if self.batch is not None:
            scores["batch"] = self.batch
        return scores


class autoHCA:
    """
    Class that is used for various cluster analysis
    Usesr provides dataframe and can afterwards
    use various metrics and methods to perfom and evaluate
    clustering.
    StandarWorkflow:
    makeLnkage() -> evalClustering() -> clusterMap() -> writeClusterFiles()
    @params
    :data:: the data to be clustered as numpy array
    :labels:: the labels for the data as numpy array
    """
    
    def __init__(self, data, clabels=None, rlabels=None, zscore=None):
    
        self.data = self._checkData(data,clabels, rlabels, zscore)
        self.rlabels = rlabels
        if clabels is not None:
            self.clabels = clabels
        else:
            self.clabels = range(1, self.data.shape[1]+1)
        self.linkage = None
        self.cluster = None
        self.clusterCol = None
        self.cmap = sns.diverging_palette(150, 275, s=80, l=55, n=9)

        
        
    @staticmethod
    def _checkData(data, clabels, rlabels, zs):
        """
        checks if data contains missing values
        if these, and the respective labels are removed
        """
        if rlabels is None and clabels is None:
            data = pd.DataFrame(data).dropna()
        
        elif rlabels is None:
            if data.shape[1] != len(clabels):
                    raise ValueError("Data and labels must be same size!")
            data = pd.DataFrame(data).dropna()
            data.columns = clabels

        elif clabels is None:
            if data.shape[0] != len(rlabels):
                raise ValueError("Data and labels must be same size!")
            temp = pd.DataFrame(data)
            to_keep = temp[temp.notnull().all(1)].index
            temp["labels"] = rlabels
            temp = temp.loc[to_keep]
            rlabels = temp["labels"]
            data = temp.drop("labels", axis=1)
            data.index = rlabels

        else:
            if data.shape[0] != len(rlabels) or data.shape[1] != len(clabels):
                raise ValueError("Data and labels must be same size!")
            temp = pd.DataFrame(data)
            to_keep = temp[temp.notnull().all(1)].index
            temp["labels"] = rlabels
            temp = temp.loc[to_keep]
            rlabels = temp["labels"]
            data = temp.drop("labels", axis=1)
            data.index = rlabels
            data.columns = clabels
            
        if zs is not None:
            temp = data.copy(deep=True).values
            tc = data.columns
            ti = data.index
            X = zscore(temp, axis=zs)
            data = pd.DataFrame(X)
            data.columns = tc
            data.index = ti
            
        return data
        
        
    def _get_N_HexCol(self, n):
        HSV_tuples = [(x*1/n, 0.75, 0.6) for x in range(n)]
        RGB_tuples = map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples)
        return list(RGB_tuples)
        
        
    def _makeCluster(self, n, colors):
        self.cluster = fcluster(self.linkage, t=n, criterion="maxclust")
        self.clusterCol = fcluster(self.linkage, t=n, criterion="maxclust")
        if colors is None or len(colors) != n:
            colors = self._get_N_HexCol(n)
        for i in range(n):
            self.clusterCol = [colors[i] if j == i+1 else j for j in self.clusterCol]
            
    
    def _makeClusterTraces(self,n,colors, z_score=None):
        plt.figure(figsize=(5,3*n))
        temp = pd.DataFrame(self.data.copy(deep=True))
        if z_score is not None:
            temp = pd.DataFrame(zscore(temp, axis=1-z_score)) #seaborn and scipy work opposit
        temp["cluster"] = self.cluster
        
        for i in range(n):
            
            ax = plt.subplot(n, 1, i+1)
            temp2 = temp[temp["cluster"]==i+1].drop("cluster", axis=1)
            
            #basically we just compute the root mean square deviation of the protein Towards the mean of the cluster
            #as a helper we take the -log of the rmsd in order To plot in the proper sequence
            temp2["distance"] = temp2.apply(lambda x: -np.log(np.sqrt(sum((x-temp2.mean())**2))),1)
            
            if temp2.shape[0] == 1:
                # if cluster contains only 1 entry
                ax.set_title("Cluster {}".format(i+1))
                ax.set_ylabel("")
                ax.set_xlabel("")
                ax.plot(range(temp2.shape[1]-1),temp2.drop("distance", axis=1).values.reshape(3), color=color[idx], alpha=alpha[idx])
                plt.xticks(range(len(self.clabels)), self.clabels)
                continue
            
            temp2["distance"] = pd.cut(temp2["distance"],5)

            #get aestethics for traces
            if colors is None:
                color=["#C72119","#D67155","#FFC288", "#FFE59E","#FFFDBF"]
            else:
                color = [colors[i]]*5
            color=color[::-1]
            alpha=[0.6,0.4,0.25,0.2,0.1]
            alpha=alpha[::-1]

            grouped = temp2.groupby("distance")

            ax.set_title("Cluster {}".format(i+1))
            ax.set_ylabel("")
            ax.set_xlabel("")
            for idx, (i, group) in enumerate(grouped):
                for j in range(group.shape[0]):
                    ax.plot(range(temp2.shape[1]-1),group.drop("distance", axis=1).iloc[j], color=color[idx], alpha=alpha[idx])

            plt.xticks(range(len(self.clabels)), self.clabels)
            
        
    def asDist(self, c):
        """
        function that takes a matrix (i.e. correlation matrix)
        and converts it into a distance matrix which can be used 
        for hierachical clustering
        @param c: correlatoin matrix
        """
        return [c[i][j] for i in (range(c.shape[0])) for j in (range(c.shape[1])) if i<j]
        
    def makeLinkage(self, method, metric):
        """
        Function that generates linkage used for HCA
        params@
        :method:: which method is used for the clustering (single, average, complete)
        :metric:: which metric is used to calculate distance
        :zscore:: whether to performe zscore transformation prior to clustering
        """
        if metric in ["pearson", "spearman"]:
            corr = pd.DataFrame(self.data).T.corr(metric).values
            dist = self.asDist(1 - corr)
        else:
            dist = distance.pdist(self.data, metric=metric)
        self.linkage = hierarchy.linkage(dist, method=method)
        
        
    def evalClustering(self,start=2, upTo=20, figsize=(15,5)):
        """
        Function evaluates number of cluster
        """
        upTo += 1
        pred = []
        for i in range(start,upTo):
            cluster = fcluster(self.linkage,t=i, criterion='maxclust')
            pred.append((davies_bouldin_score(self.data, cluster),
                       silhouette_score(self.data, cluster),
                       calinski_harabasz_score(self.data, cluster)))
                       
        pred = np.array(pred)
        plt.figure(figsize=figsize)
        plt.subplot(131)
        plt.title("Davies_boulding_score")
        plt.plot(pred[::,0])
        plt.xticks(range(upTo-start),range(start,upTo), rotation=90);
        plt.grid(axis='x')
        plt.subplot(132)
        plt.title("Silhouoette_score")
        plt.plot(pred[::,1])
        plt.xticks(range(upTo-start),range(start,upTo), rotation=90);
        plt.grid(axis='x')
        plt.subplot(133)
        plt.title("Harabasz score")
        plt.plot(pred[::,2])
        plt.xticks(range(upTo-start),range(start,upTo), rotation=90);
        plt.grid(axis='x')
        print(f"Best Davies Boulding at {start + list(pred[::,0]).index(min(pred[::,0]))} with {min(pred[::,0])}")
        print(f"Best Silhouoette_score at {start + list(pred[::,1]).index(max(pred[::,1]))} with {max(pred[::,1])}")
        print(f"Best Harabasz/Calinski at {start + list(pred[::,2]).index(max(pred[::,2]))} with {max(pred[::,2])}")
        
        
    def clusterMap(self, nCluster=None, colCluster=False, makeTraces=False, summary=False, file=None, rowColors=None,
                   colors=None, **kwargs):
        """
        :@nCluster: int; how many clusters to be generated
        :@colCluster: bool; whether to cluster the columns
        :@makeTraces: bool; whether to generated traces of each cluster
        :@summary: not implemented -> generates some kind of summary
        :@file: pathname; if provided figure is saved there
        :@roColors: dict; dictionary of title and colors for row coloring has to be same length as provided data
        """
        # summaryMap: report heatmap with means of each cluster -> also provide summary for each cluster like number of entries?
        #savemode just preliminary have to be overworked here 
        
        if nCluster is not None:
            self._makeCluster(nCluster, colors)
            
        if "cmap" not in kwargs.keys():
            kwargs["cmap"] = self.cmap
            
        if rowColors is not None:
            rowColors["cluster"] = self.clusterCol
            temp = pd.DataFrame(rowColors)
            cols = ["cluster"] + temp.drop("cluster", axis=1).columns.to_list()
            temp = temp[cols]
            temp.index = self.data.index
        else: temp = self.clusterCol

        sns.clustermap(self.data, row_linkage=self.linkage,
                   row_colors=temp, col_cluster=colCluster,
                   yticklabels="", **kwargs)
                       
        if file is not None:
            plt.savefig(file)
                       
        if makeTraces == True:
            if "z_score" in kwargs.keys():
                self._makeClusterTraces(nCluster, z_score=kwargs["z_score"], colors=colors)
            else:
                self._makeClusterTraces(nCluster, colors=colors)
            
            if file is not None:
                name, ext = file.split('.')
                filet = f"{name}_traces.{ext}"
                plt.savefig(filet)
            
           
    def returnCluster(self):
        """
        return df with clustered data
        """
        temp = self.data
        temp["cluster"] = self.cluster
        return temp
        
        
    def writeClusterFiles(self, rootdir):
        """
        generates folder with text files for each 
        cluster at provided rootdir
        """
        path = os.path.join(rootdir, "clusterResults")
        if "clusterResults" in os.listdir(rootdir):
            pass
        else:
            os.mkdir(path)
        
        temp = self.data
        temp["cluster"] = self.cluster
        for cluster in temp["cluster"].unique():
            pd.DataFrame(temp[temp["cluster"]==cluster].index).to_csv(f"{path}/cluster_{cluster}.tsv", header=None, index=False)


class KSEA:
    """
    class that performs kinase substrate enrichment analysis. You have to provide phosphoproteomic data.
    This data has to contain information about Gene name, position and amino acid of the peptides with
    "Gene names", "Position" and "Amino acid" as the respective column names. Optionally you can provide
    a "Multiplicity" column.
    """
    def __init__(self, data):
        with resources.open_text("autoprot.data","Kinase_Substrate_Dataset") as d:
            self.PSP_KS = pd.read_csv(d, sep='\t')
        self.PSP_KS["SUB_GENE"] = self.PSP_KS["SUB_GENE"].fillna("NA").apply(lambda x: x.upper())
        self.PSP_KS["source"] = "PSP"
        with resources.open_text("autoprot.data","Regulatory_sites") as d:
            self.PSP_regSits = pd.read_csv(d, sep='\t')
        self.data = self.__preprocess(data.copy(deep=True))
        self.annotDf = None
        self.kseaResults = None
        self.koi = None
        self.simpleDf = None


    @staticmethod
    def __preprocess(data):
        data["MOD_RSD"] = data["Amino acid"] + data["Position"].fillna(0).astype(int).astype(str)
        data["ucGene"] = data["Gene names"].fillna("NA").apply(lambda x: x.upper())
        data["mergeID"] = range(data.shape[0])
        return data


    def __enrichment(self,df, col, kinase):
        KS = df[col][df["KINASE"].fillna('').apply(lambda x: kinase in x)]
        s = KS.mean()#mean FC of kinase subs
        p = df[col].mean()#mean FC of all
        m = KS.shape[0]#no of kinase subs
        sig = df[col].std()#standard dev of FC of all
        score = ((s-p)*np.sqrt(m))/sig
        return [kinase,score]


    def __extractKois(self,df):
        koi = [i for i in list(df["KINASE"].values.flatten()) if isinstance(i,str) ]
        ks = set(koi)
        temp = []
        for k in ks:
            temp.append((k,koi.count(k)))
        return pd.DataFrame(temp, columns=["Kinase", "#Subs"])


    def addSubstrate(self, kinase, substrate, subModRsd):
        """
        function that allows user to manually add substrates
        
        """
        it = iter([kinase, substrate, subModRsd])
        the_len = len(next(it))
        if not all(len(l) == the_len for l in it):
             raise ValueError('not all lists have same length!')
             
        temp = pd.DataFrame(columns=self.PSP_KS.columns)
        for i in range(len(kinase)):
            temp.loc[i, "KINASE"] = kinase[i]
            temp.loc[i, "SUB_GENE"] = substrate[i]
            temp.loc[i, "SUB_MOD_RSD"] = subModRsd[i]
            temp.loc[i, "source"] = "manual"
        self.PSP_KS = self.PSP_KS.append(temp, ignore_index=True)


    def removeManualSubs(self):
        self.PSP_KS = self.PSP_KS[self.PSP_KS["source"]=="PSP"]


    def annotate(self,
        organism="human",
        onlyInVivo=False):
        """
        do we need crossspecies annotation?
        """
        
        
        if onlyInVivo==True:
            temp = self.PSP_KS[((self.PSP_KS["KIN_ORGANISM"] == organism) &
            (self.PSP_KS["SUB_ORGANISM"] == organism) & 
            (self.PSP_KS["IN_VIVO_RXN"]=="X")) | (self.PSP_KS["source"]=="manual")]
        else:
            temp = self.PSP_KS[((self.PSP_KS["KIN_ORGANISM"] == organism) &
            (self.PSP_KS["SUB_ORGANISM"] == organism)) | (self.PSP_KS["source"]=="manual")]
        
        if "Multiplicity" in self.data.columns:
            self.annotDf = pd.merge(self.data[["ucGene", "MOD_RSD", "Multiplicity","mergeID"]],
            temp,
            left_on=["ucGene", "MOD_RSD"],
            right_on=["SUB_GENE", "SUB_MOD_RSD"],
            how="left")
        else:
            self.annotDf = pd.merge(self.data[["ucGene", "MOD_RSD", "mergeID"]],
            temp,
            left_on=["ucGene", "MOD_RSD"],
            right_on=["SUB_GENE", "SUB_MOD_RSD"],
            how="left")

        
        self.koi = self.__extractKois(self.annotDf)


    def getKinaseOverview(self, kois=None):
    
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15,5))
        
        sns.histplot(self.koi["#Subs"], bins=50, ax= ax[0])
        sns.despine(ax=ax[0])
        ax[0].set_title("Overview of #Subs per kinase")

        #get axis[1] ready - basically remove everthing
        ax[1].spines["left"].set_visible(False)
        ax[1].spines["top"].set_visible(False)
        ax[1].spines["bottom"].set_visible(False)
        ax[1].spines["right"].set_visible(False)
        ax[1].tick_params(
        axis='both',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        left=False,
        labelbottom=False,
        labelleft=False) # labels along the bottom edge are off
        ax[1].set_xlim(0,1)
        ax[1].set_ylim(0,1)
        
        #plot table
        ax[1].text(x=0, y=1-0.01, s="Top10\nKinase")
        ax[1].text(x=0.1, y=1-0.01, s="#Subs")

        ax[1].plot([0,0.2], [.975,.975], color="black")
        ax[1].plot([0.1,0.1], [0,.975], color="black")

        #get top 10 kinases for annotation
        text = self.koi.sort_values(by="#Subs", ascending=False).iloc[:10].values
        for j,i in enumerate(text):
            j+=1
            ax[1].text(x=0, y=1-j/10, s=i[0])
            ax[1].text(x=0.125, y=1-j/10, s=i[1])
        #plot some descriptive stats
        tot = self.koi.shape[0]
        s = f"Substrates for {tot} kinases found in data."
        ax[1].text(0.3, 0.975, s)
        med = round(self.koi['#Subs'].median(), 2)
        s = f"Median #Sub: {med}"
        ax[1].text(0.3,0.925, s)
        mea = round(self.koi['#Subs'].mean(), 2)
        s = f"Mean #Sub: {mea}"
        ax[1].text(0.3,0.875, s)
        #if kois are provided plot those
        if kois is not None:
            pos = .8
            for j,k in enumerate(kois):
                s = self.koi[self.koi["Kinase"].apply(lambda x: x.upper()) == k.upper()]["#Subs"].values[0]
                ss = f"{k} has {s} substrates."
                ax[1].text(0.3,pos, ss)
                pos-=0.055


    def ksea(self, col, minSubs = 5, simplify=None):
        """
        function in which enrichment score is calculated
        """
        copyAnnotDf = self.annotDf.copy(deep=True)
        if simplify is not None:
            if simplify == "auto":
                simplify = {"AKT" : ["Akt1", "Akt2", "Akt3"],
                           "PKC" :  ["PKCA", "PKCD", "PKCE"],
                           "ERK" :  ["ERK1", "ERK2"],
                           "GSK3" : ["GSK3B", "GSK3A"],
                           "JNK" :  ["JNK1", "JNK2", "JNK3"],
                           "FAK" :  ["FAK iso2"],
                           "p70S6K":["p70S6K", "p70SKB"],
                           "RSK" : ["p90RSK", "RSK2"]}
            for key in simplify:
                copyAnnotDf["KINASE"].replace(simplify[key], [key]*len(simplify[key]), inplace=True)
            
            #drop rows which are now duplicates
            if "Multiplicity" in copyAnnotDf.columns:
                idx = copyAnnotDf[["ucGene","MOD_RSD","Multiplicity", "KINASE"]].drop_duplicates().index
            else:
                idx = copyAnnotDf[["ucGene","MOD_RSD", "KINASE"]].drop_duplicates().indx
            copyAnnotDf = copyAnnotDf.loc[idx]
            self.simpleDf = copyAnnotDf
            
            self.koi = self.__extractKois(self.simpleDf)
            
        koi = self.koi[self.koi["#Subs"]>=minSubs]["Kinase"]
        
        self.kseaResults = pd.DataFrame(columns = ["kinase", "score"])
        copyAnnotDf = copyAnnotDf.merge(self.data[[col,"mergeID"]], on="mergeID", how="left")
        for kinase in koi:
            k, s = self.__enrichment(copyAnnotDf[copyAnnotDf[col].notnull()], col, kinase)
            temp = pd.DataFrame(data={"kinase":k, "score":s}, index=[0])
            self.kseaResults = self.kseaResults.append(temp, ignore_index=True)
        self.kseaResults = self.kseaResults.sort_values(by="score", ascending=False)


    def returnEnrichment(self):
        if self.kseaResults is None:
            print("First perform the enrichment")
        else:
        #dropna in case of multiple columns in data
        #sometimes there are otherwise nan 
        #nans are dropped in ksea enrichment
            return self.kseaResults.dropna()


    def plotEnrichment(self, up_col="orange", down_col="blue", 
    bg_col = "lightgray", plotBg = True, ret=False, title="",
    figsize=(5,10)):
        """
        function that can be used to plot the KSEA results
        """
        if self.kseaResults is None:
            print("First perform the enrichment")
        else:
            self.kseaResults["color"] = bg_col
            self.kseaResults.loc[self.kseaResults["score"]>2, "color"] = up_col
            self.kseaResults.loc[self.kseaResults["score"]<-2, "color"] = down_col
            fig = plt.figure(figsize=figsize)
            plt.yticks(fontsize=10)
            plt.title(title)
            if plotBg == True:
                sns.barplot(data= self.kseaResults.dropna(), x="score",y="kinase", 
                palette=self.kseaResults.dropna()["color"])
            else:
                sns.barplot(data= self.kseaResults[self.kseaResults["color"]!=bg_col].dropna(), x="score",y="kinase",
                palette=self.kseaResults[self.kseaResults["color"]!=bg_col].dropna()["color"])
            sns.despine()
            plt.legend([],[], frameon=False)
            plt.axvline(0,0,1, ls="dashed", color="lightgray")
            if ret == True:
                plt.tight_layout()
                return fig


    def volcanos(self, logFC, p, kinases=[], **kwargs):
        """
        function that can be used to plot volcanos highlighting substrates
        of given kinase
        """
        df = self.annotateDf(kinases=kinases)
        for k in kinases:
            idx = df[df[k]==1].index
            vis.volcano(df, logFC, p=p, highlight=idx,
            custom_hl={"label":k},
            custom_fg={"alpha":.5}, **kwargs)


    def returnKinaseSubstrate(self, kinase):
        """
        returns new dataframe with respective substrates of kinase
        """
        if self.simpleDf is not None:
            df = self.simpleDf.copy(deep=True)
        else:
            df = self.annotDf.copy(deep=True)
            
        if isinstance(kinase, list):
            idx = []
            for k in kinase:
                idx.append(df[df["KINASE"].fillna("NA").apply(lambda x: x.upper())==k.upper()].index)
                
            dfFilter = df.loc[pl.flatten(idx)]
            
        elif isinstance(kinase, str):
            dfFilter = df[df["KINASE"].fillna("NA").apply(lambda x: x.upper())==kinase.upper()]
        dfFilter = pd.merge(dfFilter[['GENE', 'KINASE','KIN_ACC_ID','SUBSTRATE','SUB_ACC_ID',
                                       'SUB_GENE','SUB_MOD_RSD', 'SITE_GRP_ID','SITE_+/-7_AA', 'DOMAIN', 'IN_VIVO_RXN', 'IN_VITRO_RXN', 'CST_CAT#',
                                       'source', "mergeID"]],
                            self.PSP_regSits[['SITE_GRP_ID','ON_FUNCTION', 'ON_PROCESS', 'ON_PROT_INTERACT',
                                              'ON_OTHER_INTERACT', 'PMIDs', 'LT_LIT', 'MS_LIT', 'MS_CST',
                                              'NOTES']],
                                              how="left")
        return dfFilter


    def annotateDf(self, kinases=[]):
        """
        adds column to provided dataframe with given kinases and boolean value
        indicating whether or not peptide is kinase substrate
        """
        if len(kinases) > 0:
            df = self.data.drop(["MOD_RSD", "ucGene"], axis=1)
            for kinase in kinases:
                ids = self.returnKinaseSubstrate(kinase)["mergeID"]
                df[kinase] = 0
                df.loc[df["mergeID"].isin(ids),kinase] = 1
            return df.drop("mergeID", axis=1)
        else: 
            print("Please provide Kinase for annotation.")


def missAnalysis(df,cols,n=999, sort='ascending',text=True, vis=True, extraVis=False):

    """
    function takes dataframe and
    prints missing stats
    :n:: int,  how much of the dataframe shall be displayed
    :sort:: (ascending, descending) which way to sort the df
    :text:: boolean, whether to output text summaryMap
    :vis:: boolean, whether to return barplot showing missingness
    :extraVis:: boolean, whether to return matrix plot showing missingness
    """
            
    def create_data(df):
        """
        takes df and output
        data for missing entrys
        """
        data = []
        for i in df:
            #len df
            n = df.shape[0]
            #how many are missing
            m = df[i].isnull().sum()
            #percentage
            p = m/n*100
            data.append([i,n,m,p])
        data = add_ranking(data)
        return data


    def add_ranking(data):
        """
        adds ranking
        """
        data = sorted(data,key=itemgetter(3))
        for idx, col in enumerate(data):
            col.append(idx)
        
        return data
        
        
    def describe(data, n, sort='ascending'):
        """
        prints data
        :n how many entries are displayed
        :sort - determines which way data is sorted [ascending; descending]
        """
        if n == 999:
            n = len(data)
        elif n>len(data):
            print("'n' is larger than dataframe!\nDisplaying complete dataframe.")
            n = len(data)
        if n<0:
            raise ValueError("'n' has to be a positive integer!")
        
        if sort == 'ascending':
            pass
        elif sort == 'descending':
            data = data[::-1]
        
        for i in range(n):
            print("{} has {} of {} entries missing ({}%).".format(data[i][0], data[i][2], 
                                                                               data[i][1], round(data[i][3],2)))
            print('-'*80)
                    
        return True


    def visualize(data, n, sort='ascending'):
        """
        visualizes the n entries of dataframe
        :sort - determines which way data is sorted [ascending; descending]
        """
        
        import pandas as pd
        import missingno as msn
        if n == 999:
            n = len(data)
        elif n>len(data):
            print("'n' is larger than dataframe!\nDisplaying complete dataframe.")
            n = len(data)
        if n<0:
            raise ValueError("'n' has to be a positive integer!")
        
        if sort == 'ascending':
            pass
        elif sort == 'descending':
            data=data[::-1]
            
        data = pd.DataFrame(data=data, columns=["Name", "tot_values","tot_miss", "perc_miss", "rank"])
        plt.figure(figsize=(7,7))
        ax = plt.subplot()
        splot = sns.barplot(x=data["tot_miss"].iloc[:n], y=data["Name"].iloc[:n])
        for idx,p in enumerate(splot.patches):
            s=str(round(data.iloc[idx,3],2))+'%'
            x=p.get_width()+p.get_width()*.01
            y=p.get_y()+p.get_height()/2
            splot.annotate(s, (x,y))
        plt.title("Missing values of dataframe columns.")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        return True


    def visualize_extra(df):
        import missingno as msn
        plt.figure(figsize=(10,7))
        msn.matrix(df, sort="ascending")
        return True
    
    
    df = df[cols]
    data = create_data(df)
    if text==True:
        describe(data,n, sort)
    if vis == True:
        visualize(data,n, sort)
    if extraVis == True:
        visualize_extra(df)
        

def getPubAbstracts(text=[""], ToA=[""], author=[""], phrase=[""], 
                  exclusion=[""], customSearchTerm=None, makeWordCloud=False,
                  sort='pub+date', retmax=20, output="print"):
    """
    Get Pubmed abstracts
    Provide the search terms and pubmed will be searched for paper matching those
    The exclusion list can be used to exclude certain words to be contained in a paper
    If you want to perform more complicated searches use the possibility of using
    a custom term. 
    
    If you search for a not excact term you can use *
    For example opto* will also return optogenetic.
    
    @params:
    ::ToA: List of words which must occur in Titel or Abstract
    ::text: List of words which must occur in text
    ::author: List of authors which must occor in Author list
    ::phrase: List of phrases which should occur in text. Seperate word in phrases with hyphen (e.g. akt-signaling)
    ::exclusion: List of words which must not occur in text
    ::CustomSearchTerm: Enter a custom search term (make use of advanced pubmed search functionality)
    ::makeWordCloud: Boolen, toggles whether to draw a wordcloud of the words in retrieved abstrace
    """
    
    

    def search(query,retmax, sort):
        Entrez.email = 'your.email@example.com'
        handle = Entrez.esearch(db='pubmed',
                                sort=sort,
                                retmax=str(retmax),
                                retmode='xml',
                                term=query)
        results = Entrez.read(handle)
        return results


    def fetchDetails(id_list):
        ids = ','.join(id_list)
        Entrez.email = 'your.email@example.com'
        handle = Entrez.efetch(db='pubmed',
                               retmode='xml',
                               id=ids)
        results = Entrez.read(handle)
        return results


    def makeSearchTerm(text, ToA, author, phrase, exclusion):

        term = [i+"[Title/Abstract]" if i != "" else "" for i in ToA] +\
                [i+"[Text Word]" if i != "" else "" for i in text] +\
                [i+"[Author]" if i != "" else "" for i in author] + \
                [i if i != "" else "" for i in phrase]
        term = (" AND ").join([i for i in term if i != ""])

        if exclusion != [""]:
            exclusion = (" NOT ").join(exclusion)
            term = (" NOT ").join([term] + [exclusion])
            return term
        else:
            return term


    def makewc(final, output):
        abstracts = []
        for i in final:
            for abstract in final[i][2]:
                if not isinstance(abstract, str):
                    abstracts.append(abstract["AbstractText"][0])
        abstracts = (" ").join(abstracts)
        
        #when exlusion list is added in wordcloud add those
        el = ["sup"]

        fig = vis.wordcloud(text=abstracts, exlusionwords=el)
        if output != "print":
            if output[:-4] == ".txt":
                output = output[:-4]
            fig.to_file(output + "/WordCloud.png")


    def genOutput(final, output, makeWordCloud):

        if output == "print":
            for i in final:
                for title,author, abstract, doi, link, pid, date, journal in\
                zip(final[i][0],final[i][1],final[i][2],final[i][3],final[i][4],final[i][5],final[i][6],final[i][7]):
                    print(title)
                    print('-'*100)
                    print(("; ").join(author))
                    print('-'*100)
                    print(journal)
                    print(date)
                    print(f"doi: {doi}, PubMedID: {pid}")
                    print(link)
                    print('-'*100)
                    if isinstance(abstract, str):
                        print(abstract)
                    else:
                        print(abstract["AbstractText"][0])
                    print('-'*100)
                    print('*'*100)
                    print('-'*100)
                    
        elif output[-4:] == ".txt":
            with open(output, 'w', encoding="utf-8") as f:
                for i in final:
                    for title,author, abstract, doi, link, pid, date, journal in\
                    zip(final[i][0],final[i][1],final[i][2],final[i][3],final[i][4],final[i][5],final[i][6],final[i][7]):
                        f.write(title)
                        f.write('\n')
                        f.write('-'*100)
                        f.write('\n')
                        f.write(("; ").join(author))
                        f.write('\n')
                        f.write('-'*100)
                        f.write('\n')
                        f.write(journal)
                        f.write('\n')
                        f.write(str(date))
                        f.write('\n')
                        f.write(f"doi: {doi}, PubMedID: {pid}")
                        f.write('\n')
                        f.write(link)
                        f.write('\n')
                        f.write('-'*100)
                        f.write('\n')
                        writeAbstract(abstract, f, "txt")
                        f.write('\n')
                        f.write('-'*100)
                        f.write('\n')
                        f.write('*'*100)
                        f.write('\n')
                        f.write('-'*100)
                        f.write('\n')
                        
        # would be nice to have a nicer format -> idea write with html and save as such! :) noice!
        else:
            if output[-5:] == ".html":
                pass 
            else:
                output += "/PubCrawlerResults.html"
            with open(output, 'w', encoding="utf-8") as f:
                f.write("<!DOCTYPE html>")
                f.write("<html>")
                f.write("<head>")
                f.write("<style>")
                f.write(".center {")
                f.write("display: block;")
                f.write("margin-left: auto;")
                f.write("margin-right: auto;")
                f.write("width: 80%;}")
                f.write("</style>")
                f.write("</head>")
                f.write('<body style="background-color:#FFE5B4">')
                if makeWordCloud == True:
                    f.write(f'<img src="WordCloud.png" alt="WordCloud" class="center">')
                for i in final:
                    for title,author, abstract, doi, link, pid, date, journal in\
                    zip(final[i][0],final[i][1],final[i][2],final[i][3],final[i][4],final[i][5],final[i][6],final[i][7]):
                        f.write(f"<h2>{title}</h2>")
                        f.write('<br>')
                        ta = ("; ").join(author)
                        f.write(f'<p style="color:gray; font-size:16px">{ta}</p>')
                        f.write('<br>')
                        f.write('-'*200)
                        f.write('<br>')
                        f.write(f"<i>{journal}</i>")
                        f.write('<br>')
                        f.write(str(date))
                        f.write('<br>')
                        f.write(f"<i>doi:</i> {doi}, <i>PubMedID:</i> {pid}")
                        f.write('<br>')
                        f.write(f'<a href={link}><i>Link to journal</i></a>')
                        f.write('<br>')
                        f.write('-'*200)
                        f.write('<br>')
                        writeAbstract(abstract, f, "html")
                        f.write('<br>')
                        f.write('-'*200)
                        f.write('<br>')
                        f.write('*'*150)
                        f.write('<br>')
                        f.write('-'*200)
                        f.write('<br>')
                f.write("</body>")
                f.write("</html>")


    def writeAbstract(abstract, f, typ):
        if not isinstance(abstract, str):
            abstract = abstract["AbstractText"][0]
        abstract = abstract.split(" ")
                
        if typ == "txt":
            for idx, word in enumerate(abstract):
                f.write(word)
                f.write(' ')
                if idx%20==0 and idx>0:
                    f.write('\n')
                
               
        elif typ == "html":
            f.write('<p style="color:#202020">')
            for idx, word in enumerate(abstract):
                f.write(word)
                f.write(' ')
                if idx%20==0 and idx>0:
                    f.write('</p>')
                    f.write('<p style="color:#202020">')
            f.write('</p>')

    

    # if function is executed in a loop a small deleay to ensure pubmed is responding
    time.sleep(0.5)
    
    if customSearchTerm is None:
        term = makeSearchTerm(text, ToA, author, phrase,
                      exclusion)
    else:
        term = customSearchTerm
    
    results = search(term, retmax, sort)["IdList"]
    if len(results) > 0:
        results2 = fetchDetails(set(results))
    
        titles = []
        abstracts = []
        authors = []
        dois = []
        links = []
        pids = []
        dates = []
        journals = []
        final = dict()
        
        for paper in results2["PubmedArticle"]:
            #get titles
            titles.append(paper["MedlineCitation"]["Article"]["ArticleTitle"])
            #get abstract
            if "Abstract" in paper["MedlineCitation"]["Article"].keys():
                abstracts.append(paper["MedlineCitation"]["Article"]["Abstract"])
            else:
                abstracts.append("No abstract")
            #get authors
            al = paper['MedlineCitation']['Article']["AuthorList"]
            tempAuthors = []
            for a in al:
                if "ForeName" in a and "LastName" in a:
                    tempAuthors.append([f"{a['ForeName']} {a['LastName']}"])
                elif "LastName" in a:
                    tempAuthors.append([a['LastName']])
                else:
                    tempAuthors.append(a)
            authors.append(list(pl.flatten(tempAuthors)))
            #get dois and make link
            doi = [i for i in paper['PubmedData']['ArticleIdList'] if i.attributes["IdType"]=="doi"]
            if len(doi) > 0:
                doi = str(doi[0])
            else: doi = "NaN"
            dois.append(doi)
            links.append(f" https://doi.org/{doi}")
            #get pids
            pids.append(str([i for i in paper['PubmedData']['ArticleIdList'] if i.attributes["IdType"]=="pubmed"][0]))
            #get dates
            
            rawDate = paper['MedlineCitation']['Article']['Journal']['JournalIssue']['PubDate']
            try:
                date = f"{rawDate['Year']}-{rawDate['Month']}-{rawDate['Day']}"
            except:
                date = rawDate
            dates.append(date)
            #get journal
            journals.append(paper['MedlineCitation']['Article']['Journal']['Title'])
        
        final[term] = (titles,authors,abstracts,dois, links, pids, dates, journals)

        if makeWordCloud == True:
            makewc(final, output)
        genOutput(final, output, makeWordCloud)
 
    else:
        print("No results for " + term)


def loess(xvals, yvals, data, alpha, poly_degree=1):
    """
    https://medium.com/@langen.mu/creating-powerfull-lowess-graphs-in-python-e0ea7a30b17a
    example:
    
    evalDF = loess("t1/2_median_HeLa", "t1/2_median_Huh", data = comp, alpha=0.9, poly_degree=1)

    fig = plt.figure()
    ax = plt.subplot()
    sns.scatterplot(comp["t1/2_median_HeLa"], comp["t1/2_median_Huh"])

    ax.plot(evalDF['v'], evalDF['g'], color='red', linewidth= 3, label="Test")
    plt.xlim(1,12)
    plt.ylim(1,12)
    ax.plot([0, 1], [0, 1], transform=ax.transAxes)
    """
    
    def loc_eval(x, b):
        loc_est = 0
        for i in enumerate(b): loc_est+=i[1]*(x**i[0])
        return(loc_est)

    all_data = sorted(zip(data[xvals].tolist(), data[yvals].tolist()), key=lambda x: x[0])
    xvals, yvals = zip(*all_data)
    evalDF = pd.DataFrame(columns=['v','g'])
    n = len(xvals)
    m = n + 1
    q = int(np.floor(n * alpha) if alpha <= 1.0 else n)
    avg_interval = ((max(xvals)-min(xvals))/len(xvals))
    v_lb = min(xvals)-(.5*avg_interval)
    v_ub = (max(xvals)+(.5*avg_interval))
    v = enumerate(np.linspace(start=v_lb, stop=v_ub, num=m), start=1)
    xcols = [np.ones_like(xvals)]
    for j in range(1, (poly_degree + 1)):
        xcols.append([i ** j for i in xvals])
    X = np.vstack(xcols).T
    for i in v:
        iterpos = i[0]
        iterval = i[1]
        iterdists = sorted([(j, np.abs(j-iterval)) for j in xvals], key=lambda x: x[1])
        _, raw_dists = zip(*iterdists)
        scale_fact = raw_dists[q-1]
        scaled_dists = [(j[0],(j[1]/scale_fact)) for j in iterdists]
        weights = [(j[0],((1-np.abs(j[1]**3))**3 if j[1]<=1 else 0)) for j in scaled_dists]
        _, weights      = zip(*sorted(weights,     key=lambda x: x[0]))
        _, raw_dists    = zip(*sorted(iterdists,   key=lambda x: x[0]))
        _, scaled_dists = zip(*sorted(scaled_dists,key=lambda x: x[0]))
        W         = np.diag(weights)
        b         = np.linalg.inv(X.T @ W @ X) @ (X.T @ W @ yvals)
        local_est = loc_eval(iterval, b)
        iterDF2   = pd.DataFrame({
                       'v'  :[iterval],
                       'g'  :[local_est]
                       })
        evalDF = pd.concat([evalDF, iterDF2])
    evalDF = evalDF[['v','g']]
    return(evalDF)







def test():
    # challenging for the implementaiton of R subroutine is to find location of R exe 
    # (and of course version control... 
    import os
    print(os.getcwd())
# #This is for R subroutine. Maybe nice for normalization, imputation and a basic form of limma.
#command = ['C:/Program Files/R/R-3.6.1/bin/x64/Rscript.exe', '--vanilla','C:/Users/Wignand/Notebooks/Python/PyR_test/test.R',
#                  folder]
#        result = run(command, stdout=PIPE, stderr=PIPE, universal_newlines=True)
#        limma_res = feather.read_dataframe(folder+"/for_limma_out")