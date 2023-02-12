import PyPDF2, re, nltk
corpusraw = {'multivariable', 'calculus', 'linear', 'algebra', 'mathematical', 'concept', 'multivariables', 'advanced', 'eigenvalues', 'eigenvectors', 'compute', 'matrices', 'decompositions', 'evaluate', 'partial', 'derivatives', 'extrema', 'taylor', 'series', 'multivariate', 'function', 'functions', 'integrals', 'multiple', 'line', 'theorems', 'green', 'divergence', 'stokes', 'mathematical', 'computational', 'methods', 'range', 'algebra', 'analysis', 'orthogonality', 'partial', 'differentiation', 'multiple', 'integration', 'gradient', 'divergence', 'curl', 'line', 'surface', 'theorems', 'gauss', 'probability', 'statistics', 'statistical', 'interference', 'theoretical', 'distribution', 'probabilistic', 'hypothesis', 'testing', 'parametric', 'estimation', 'problems', 'formulate', 'computational', 'inference', 'sample', 'space', 'discrete', 'continuous', 'random', 'variables', 'central', 'limit', 'theorem', 'chebyshev', 'expectation', 'variances', 'moment', 'generating', 'estimation', 'parameters', 'numerical', 'python', 'applications', 'actuarial', 'science', 'engineering', 'stochastic', 'process', 'machine', 'learning', 'data', 'r', 'matlab', 'c', 'sas', 'heuristic', 'rigorous', 'simple', 'General', 'regression', 'reduction', 'logistic', 'support', 'vector', 'machine', 'gaussian', 'neural', 'network', 'model', 'algorithms', 'gradient', 'deep', 'programming', 'datasets', 'open', 'source', 'classification', 'softmax', 'biasvariance', 'tradeoff', 'regularization', 'rademacher', 'vcdimension', 'Generalization', 'error', 'estimation', 'approimation', 'kernel', 'hilbert', 'spaces', 'inequalities', 'empirical', 'risk', 'minimization', 'convex', 'optimization', 'emalgorithm', 'econometrics', 'capital', 'asset', 'pricing', 'financial', 'markets', 'computer', 'parametric', 'nonparametric', 'averaging', 'aggregation', 'highdimensional', 'data', 'modelling', 'data', 'mining', 'bayesian', 'decision', 'nonlinear', 'discriminant', 'analysis', 'clustering', 'networks', 'decision', 'trees', 'association', 'rule', 'business', 'functions', 'limits', 'continuity', 'differentiability', 'change', 'local', 'extrema', 'optimization',
             'substitution', 'parts', 'present', 'value', 'matrices', 'determinants', 'elimination', 'inverses', 'gaussjordan', 'inputoutput', 'polar', 'cartesian', 'euler', 'relation', 'dot', 'cross', 'triple', 'scalar', 'quotient', 'chain', 'leibnitz', 'region', 'curves', 'polynomials', 'induction', 'binomial', 'coordinate', 'geometry', 'conic', 'sections', 'trigonometry', 'inverses', 'limits', 'implicit', 'logarithmic', 'successive', 'concentration', 'pac', 'vc', 'dimension', 'rademacher', 'convex', 'gradient', 'decent', 'boosting', 'kernels', 'support', 'vector', 'machines', 'learning', 'bounds', 'linear', 'equation', 'equations', 'orthogonality', 'diagonalization', 'hermitian'} 
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer 
ps = PorterStemmer() 
# import nltk for stopwords
from nltk.corpus import stopwords
from scipy.spatial.distance import cosine
import numpy as np
from numpy.linalg import norm
stop_words = set(stopwords.words('english'))

def pdf_reader(pdf_file):
    string_pages = ""
    for i in range(len(pdf_file.pages)):
        # Get the current page
        page = pdf_file._get_page(i)
        # Extract the text from the page and add it to the string
        string_pages += page.extract_text()
    return string_pages

def normalizer(string_pages):
    # convert to lower case
    lower_string_pages = string_pages.lower()
    # remove numbers
    no_number_string_pages = re.sub(r'\d+','',lower_string_pages)
    # remove all punctuation except words and space
    no_punc_string_pages = re.sub(r'[^\w\s]','', no_number_string_pages)
    # remove white spaces
    no_wspace_string_pages = no_punc_string_pages.strip()
    no_wspace_string_pages
    # convert string to list of words
    lst_string_pages = [no_wspace_string_pages][0].split()
    # remove stopwords
    no_stpwords_string_pages=""
    for i in lst_string_pages:
        if not i in stop_words:
            no_stpwords_string_pages += i+' '
    # removing last space
    no_stpwords_string_pages = no_stpwords_string_pages[:-1]
    clean_list= [i for i in no_stpwords_string_pages.split() if len(i)>2]
    #only want words in corpus)
    final_list=set(i for i in clean_list if i in corpusraw)
    return final_list

file1 = open("CityU_MA3526.pdf", "rb")
pdf_file1 = PyPDF2.PdfReader(file1)

file2 = open("McGill_MATH316.pdf", "rb")
pdf_file2 = PyPDF2.PdfReader(file2)

#defining variables
s1= normalizer(pdf_reader(pdf_file1))
s2= normalizer(pdf_reader(pdf_file2))



# Python3 implementation of the approach
 
# Function to return the
# intersection set of s1 and s2
def intersection(s1, s2) :
 
    # Find the intersection of the two sets
    intersect = s1 & s2 ;
    return intersect;
 
 
# Function to return the Jaccard index of two sets
def jaccard_index(s1, s2) :
     
    # Sizes of both the sets
    size_s1 = len(s1);
    size_s2 = len(s2);
 
    # Get the intersection set
    intersect = intersection(s1, s2);
 
    # Size of the intersection set
    size_in = len(intersect);
 
    # Calculate the Jaccard index
    # using the formula
    jaccard_in = size_in  / (size_s1 + size_s2 - size_in);
 
    # Return the Jaccard index
    return jaccard_in;
 
 
# Function to return the Jaccard distance
def jaccard_distance(jaccardIndex)  :
 
    # Calculate the Jaccard distance
    # using the formula
    jaccard_dist = 1 - jaccardIndex;
 
    # Return the Jaccard distance
    return jaccard_dist;
 
 
jaccardIndex = jaccard_index(s1, s2);

import math
from collections import Counter

def cosine_similarity(doc1, doc2):
    
    # Get the word count for each document
    doc1_word_counts = Counter(doc1)
    doc2_word_counts = Counter(doc2)
    
    # Get the unique words in each document
    all_words = set(doc1_word_counts.keys()).union(set(doc2_word_counts.keys()))
    
    # Create a vector for each document where the value in each dimension is the count of the word in the corresponding document
    doc1_vector = [doc1_word_counts.get(word, 0) for word in all_words]
    doc2_vector = [doc2_word_counts.get(word, 0) for word in all_words]
    
    # Calculate the dot product of the two vectors
    dot_product = sum([x * y for x, y in zip(doc1_vector, doc2_vector)])
    
    # Calculate the magnitude (length) of each vector
    doc1_magnitude = math.sqrt(sum([x**2 for x in doc1_vector]))
    doc2_magnitude = math.sqrt(sum([x**2 for x in doc2_vector]))
    
    # Calculate the cosine similarity
    cosine_similarity = dot_product / (doc1_magnitude * doc2_magnitude)
    
    return cosine_similarity

 
# Print the Jaccard index and Jaccard distance
print("Jaccard index = ",jaccardIndex);
print("Jaccard distance = ",jaccard_distance(jaccardIndex));
print("Cosine similarity = ",cosine_similarity(s1,s2))

import os
import pandas as pd


def compare_files(dir1, dir2):
    # Get the list of files in each directory
    files1 = os.listdir(dir1)
    files2 = os.listdir(dir2)

    results = pd.DataFrame(columns=['cityu', 'edin', 'Similarity Index Jaccard', 'Similarity Index Cosine'])
    # Loop through each file and compare the contents
    for i in range(len(files1)):
        for j in range(len(files2)):
            file1 = os.path.join(dir1, files1[i])
            file2 = os.path.join(dir2, files2[j])

            with open(file1, 'rb') as f1, open(file2, 'rb') as f2:
                pdf_file1 = PyPDF2.PdfReader(file1)
                pdf_file2 = PyPDF2.PdfReader(file2)

                s1= normalizer(pdf_reader(pdf_file1))
                s2= normalizer(pdf_reader(pdf_file2))

                results = results.append({'cityu': os.path.basename(file1), 'edin': os.path.basename(file2), 'Similarity Index Jaccard': jaccard_index(s1, s2), 'Similarity Index Cosine': cosine_similarity(s1,s2)}, ignore_index=True)

    return results.to_excel("results_cityu_edin2.xlsx", index=False)

results = pd.DataFrame(columns=['cityu', 'edin', 'Similarity Index Jaccard', 'Similarity Index Cosine'])
results.to_excel('results_cityu_edin2.xlsx', index=False)
#compare_files('src\cityu', 'src\edin')
