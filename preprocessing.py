# -*- coding: utf-8 -*-
"""

Created on Mon Jul  8 09:26:07 2019

@author: Wignand

DataProcessing

:function cleaning: for first processing of dataframe ratio cols
"""

import numpy as np
import pandas as pd
from importlib import resources
import re
from autoprot.decorators import report

def read_csv(file, sep='\t'):
    return pd.read_csv(file, sep=sep)
    
    
def to_csv(df, file, sep='\t', index=False):
    df.to_csv(file, sep=sep, index=index)

@report
def cleaning(df, file="proteinGroups"):
    """
    removes contaminant, reverse and identified by site only entries
    @file:: which file is provided:
        proteinGroups; Phospho (STY); evidence; 
        modificationSpecificPeptides 
    """
    columns = df.columns
    if file == "proteinGroups":
        if ("Potential contaminant" not in columns) or\
        ("Reverse" not in columns) or\
        ("Only identified by site" not in columns):
            print("Is this data already cleaned?\nMandatory columns for cleaning not present in data!")
            print("Returning provided dataframe!")
            return df
        df = df[(df['Potential contaminant'].isnull()) &
               (df['Reverse'].isnull()) &
               (df['Only identified by site'].isnull())]
        df.drop(['Potential contaminant',"Reverse", 'Only identified by site'], axis=1, inplace=True)
    elif (file == "Phospho (STY)") or (file == "evidence") or (file == "modificationSpecificPeptides"):
        if ("Potential contaminant" not in columns) or\
        ("Reverse" not in columns):
            print("Is this data already cleaned?\nMandatory columns for cleaning not present in data!")
            print("Returning provided dataframe!")
            return df
        df = df[(df['Potential contaminant'].isnull()) &
               (df['Reverse'].isnull())]
        df.drop(['Potential contaminant',"Reverse"], axis=1, inplace=True)
    return df


def log(df, cols, base=2, invert=None):
    """
    performs log transformation. Returns dataframe with additional log columns
    @params
    ::cols: cols which are transformed
    ::base: base of log, default=2, alternative: 10
    ::invert: vector corresponding to columns telling which to invert
    
    """
    if base == 2:
      for c in cols:
            df[f"log2_{c}"] = np.log2(df[c])
    elif base==10:
        for c in cols:
            df[f"log10_{c}"] = np.log10(df[c])
    else:
        print("This base is not implemented!")
    if invert is not None:
        lcols = df.filter(regex="^log").columns
        df[lcols] = df[lcols] * invert
    return df


def locProts(df, thresh=.75):
    """
    removes entries with localiatoin probabiliy below threshold
    @params
    @df :: dataframe to be filtered
    @thresh :: threshold of localization probability
    """
    if "Localization prob" not in df.columns:
        print("This dataframe has no 'Localization prob' column!")
        return True
    print(f"{df.shape[0]} entries in dataframe.")
    df = df[df["Localization prob"]>=thresh]
    print(f"{df.shape[0]} entries in dataframe with localization prob >= {thresh*100}%.")
    return df

@report
def removeNonQuant(df, cols):
    """
    removes entries without quantitative data
    @params
    @df :: dataframe to be filtered
    @cols :: cols to be evaluated for missingness
    """
    df = df[~(df[cols].isnull().all(1))]
    return df


def expandSiteTable(df, cols):
    """
    function that expands the phosphosite table Sites -> peptides
    x, a__1, a__2, a__3
    ->
    x, a, 1
    x, a, 2
    x, a, 3
    @params
    @df :: dataframe to be expanded (important that an "id" column is provided)
    @cols :: cols which are going to be expanded (format: Ratio.*___.)
    """
    print(f"{df.shape[0]} phosphosites in dataframe.")
    dfs = []
    expected = df.shape[0]*3
    #columns to melt
    melt = cols
    melt_set = list(set([i[:-4] for i in melt]))
    #Due to MaxQuant column names we might have to drop some columns
    check = [i in df.columns for i in melt_set]
    if False not in check:
        df.drop(melt_set, axis=1, inplace=True)
    if True in check and False in check:
        print("Your provided columns ")
        raise ValueError("The columns you provided are not suitable!")
    for i in melt_set:
        cs = list(df.filter(regex=i+'___').columns )+ ["id"]
        dfs.append(pd.melt(df[cs], id_vars='id'))
    temp = df.copy(deep=True)
    temp = temp.drop(melt, axis=1)
    
    for idx,df in enumerate(dfs):
        x = df["variable"].iloc[0].split('___')[0]
        if idx==0:
            t = df.copy(deep=True)
            t.columns = ["id", "Multiplicity", x]
            t["Multiplicity"] = t["Multiplicity"].apply(lambda x: x.split('___')[1])
        else:
            df.columns = ["id", "Multiplicity", x]
            df = df.drop(["id", "Multiplicity"], axis=1)
            t = t.join(df,rsuffix=idx)
    temp = temp.merge(t,on='id', how='left')
    if temp.shape[0] != expected:
        print("The expansion of site table is probably not correct!!! Check it! Maybe you provided wrong columns?")
    temp = temp[~(temp[melt_set].isnull().all(1))]
    print(f"{temp.shape[0]} phosphopeptides in dataframe after expansion.")
    return temp

@report
def filterVv(df, groups,n=2, vv=True):
    """
....function that filters dataframe for minimum number of valid values
....@params
    df :: dataframe to be filtered - copy is returned
    groups :: the experimental groups. Each group is filtered for at least n vv
    n :: minimum amount of valid values
    vv :: True for minimum amount of valid values; False for maximum amount of missing values
...."""
    if vv == True:
        idxs = [set(df[(len(group)-df[group].isnull().sum(1)) >= n].index) for\
               group in groups]
    else:
        idxs =  [set(df[df[group].isnull().sum(1) <= n].index) for\
               group in groups]

    #take intersection of idxs
    idx = set.intersection(*idxs)
    df = df.loc[idx]
    return df


def GoAnnot(prots, gos, onlyProts=False):

    """
    function that finds kinases based on go annoation in 
    list of gene names. If there are multiple gene names separated by semicolons
    only the first entry will be used.
    :@Prots: List of Gene names
    :@go: List of go terms
    Notes:
        Homo sapiens.gene_info and gene2go files 
        are needed for annotation
        
        In case of multiple gene names per line (e.g. AKT1;PKB)
        only the first name will be extracted.
    """
    with resources.open_text("autoprot.data","Homo_sapiens.gene_info") as d:
        geneInfo = pd.read_csv(d, sep='\t')
    with resources.open_text("autoprot.data","gene2go_alt") as d:
        gene2go = pd.read_csv(d, sep='\t')
    prots = pd.DataFrame(pd.Series([str(i).upper().split(';')[0] for i in prots]), columns=["Gene names"])
    prots = prots.merge(geneInfo[["Symbol", "GeneID"]], left_on="Gene names", right_on="Symbol", how='inner')
    
    prots = prots.merge(gene2go[["GeneID", "GO_ID", "GO_term"]], on="GeneID", how='inner')
    if onlyProts == True:
        for idx, go in enumerate(gos):
            if idx == 0:
                redProts = prots["Symbol"][prots["GO_term"].str.contains(go)]
            else:
                redProts = redProts.append(prots["Symbol"][prots["GO_term"].str.contains(go)])
        return redProts.drop_duplicates()
    else: 
        for idx, go in enumerate(gos):
            if idx == 0:
                redProts = prots[prots["GO_term"]==go]
            else:
                redProts = redProts.append(prots[prots["GO_term"]==go])
        return redProts.drop_duplicates()


def motifAnnot(df, motif, col=None):
    """
    Function that searches for phosphorylation motif in the provided dataframe.
    If not specified "Sequence window" column is searched. Phosphorylated central residue
    has to indicated with S/T, arbitrary amino acids with x. 
    Examples:
    - RxRxxS/T
    - PxS/TP
    - RxRxxS/TxSxxR

    :@df: dataframe
    :@motif: str; motif to be searched for
    :@col: str; alternative column to be searched in if Sequence window is not desired
    """

    #make some assertions that the column is indeed the proper MQ output
    #(might want to customize the possibilites later)
    
    def findMotif(x,col, motif, motlen):
        seq = x[col]
        if ";" in seq:
            seqs = seq.split(';')
        else: seqs = [seq]
        for seq in seqs:
            pos = 0
            pos2 = re.finditer(motif,seq)
            if pos2:
                for p in pos2:
                    pos = p.end()
                    if pos == np.floor(motlen/2+1):
                        return 1
        return 0
    
    if col is None:
        col = "Sequence window"
    
    assert(col in df.columns)
    assert(len(df[col].iloc[0]) % 2 == 1)
    
    
    
    search = motif.replace('x', '.').replace('S/T', '(S|T)').upper()
    i = search.index("(S|T)")
    before = search[:i]
    after  = search[i+5:]
    search = f"(?<={before})(S|T)(?={after})"
    motlen = len(df[col].iloc[0])
    df[motif] = df.apply(findMotif, col=col, motif=search, motlen=motlen, axis=1)
    
    return df


