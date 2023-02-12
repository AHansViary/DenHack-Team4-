from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
import PyPDF2, re, nltk
corpusraw = {'general', 'fixation', 'skew', 'height', 'guidelines', 'side', 'subset', 'verify', 'distributing', 'parametrize', 'vertex', 'displacement', 'generalization', 'inequality', 'sequences', 'variables', 'paraboloid', 'seconddegree', 'isolate', 'above','cuboid', 'differentiation', 'variance', 'acceptance', 'inclination', 'rationalizing', 'delta', 'partition', 'quadrangle', 'cube', 'association', 'event', 'iterative', 'maximum', 'harmonic', 'size', 'counterclockwise', 'arithmetic', 'problems', 'even', 'sequence', 'contig', 'associative', 'contrapositive', 'graphic', 'intervals', 'software', 'identification', 'signs', 'density', 'particular', 'pascal', 'patterns', 'quintuple', 'column', 'direction', 'invention', 'transition', 'transversion', 'initial', 'times', 'factorization', 'integer', 'isometry', 'tabular', 'integrable', 'expansion', 'parts', 'rational','sections', 'consistency', 'improper', 'irrational','transitive', 'statistics', 'precise', 'trigonometry', 'programming', 'acute','term', 'multinomial', 'algebraic', 'differentiable','phi', 'primer', 'equidistant', 'sigma','curvesketching', 'recurrence', 'empirical', 'conchoid', 'geometry', 'periodic', 'algebra', 'structure', 'radian', 'compounding', 'unit', 'component', 'equivalent','disk','univariate', 'semicircle', 'rhombus', 'slopeintercept', 'map', 'calculation', 'dimensions', 'dispute', 'mutually', 'relationship', 'fraction', 'net', 'axis',
'setbuilder', 'implications', 'error', 'degree', 'logarithmic', 'independent', 'region','volume', 'count', 'curves', 'inconsistent', 'octagon', 'square', 'integration','solutions', 'isosceles', 'diagrams', 'runs', 'quantitative', 'clustering', 'fractions','transcendental', 'trapezoid', 'lambda', 'effect', 'indeterminate', 'negative', 'complement', 'arc', 'poisson', 'traversal', 'values', 'defining', 'abscissa', 'directly', 'cavalieris', 'space', 'felsensteins', 'coordinate', 'chain', 'waiting', 'dimensional', 'impulse', 'lengths', 'centers', 'stilescrawford', 'quasinewton', 'present', 'distal', 'applications', 'penetrance', 'regression', 'lub', 'incircle', 'spherical', 'ratios', 'circumcircle', 'diagonalization', 'dirichlet', 'quartile', 'nthdegree', 'heaviside', 'googolplex', 'heritability', 'best', 'equations', 'quintiles', 'concave', 'randomization', 'factorial', 'cot', 'construction', 'proximal','numbering', 'detection', 'experiment', 'stochastic', 'discontinuity', 'reversible', 'machine', 'trapezium', 'combinations', 'position', 'relation', 'conditionally', 'null', 'equiangular', 'trig', 'limit', 'assumptions', 'progression', 'polyhedron', 'logarithm', 'graphically', 'reflection', 'angles', 'hypotheses', 'shift', 'annulus', 'ifthen', 'percentage', 'oval', 'amplitude', 'trajectories', 'midpoint', 'equilibrium', 'tangent', 'hexagon', 'array', 'upsilon', 'listing', 'satisfy','admixture', 'asymptote', 'cosec', 'fitting', 'correlation', 'deviation', 'convergence', 'asymptotic', 'function', 'crosssection', 'lhospital', 'denominator', 'quartiles', 'model', 'zero', 'continuity', 'pair', 'clockwise', 'nonzero', 'interal', 'orbits', 'shape', 'increasing', 'spread', 'conditional', 'deleted', 'multiplicative', 'ordinate', 'parameters', 'transversal', 'logarithms', 'rolles', 'derivatives', 'minimize', 'directrices', 'backtracking', 'group', 'bac', 'yaxis', 'arithmeticgeometric', 'paired', 'normalizing','continued', 'intercept', 'cylinder', 'cone', 'chains', 'approximation', 'scoring', 'metaphase', 'cylindrical', 'parabola', 'sss', 'major', 'graph', 'overdetermined', 'circumscribed', 'mean', 'least', 'plot', 'ratio', 'monomial', 'quotient', 'diverge', 'abo', 'reflected', 'multiplier', 'csc', 'transformation', 'ctg', 'toolkit', 'success', 'calculus', 'rectangular', 'phase', 'underdetermined', 'concentration', 'planes', 'commutative', 'undefined', 'consistent', 'euclidean', 'alignment', 'approximating', 'symmetry', 'continuously', 'focal', 'sines', 'repeated', 'factors', 'indicators', 'determining', 'functions', 'minor', 'exponential', 'noneuclidean', 'logconcavity', 'subtraction', 'matrix', 'sampling', 'coplanar', 'components', 'segment', 'apex', 'twointercept', 'random', 'triangulation', 'formula', 'fractional', 'row', 'ascertainment', 'differentiability', 'rare', 'state', 'stretched', 'extraneous', 'algorithm', 'xaxis', 'nonconvex', 'arctan', 'inradius', 'estimation', 'techniques', 'circles',  'multiplying', 'arcsine', 'symmetric', 'flip', 'equivalence', 'integrating',  'xcoordinate', 'cosines', 'counting', 'estimating', 'hypotenuse', 'compute',  'point', 'maximize', 'exclusion', 'iqr', 'qed', 'elliptic', 'second', 'antiderivatives', 'tests', 'covariances', 'normal', 'disjoint', 'distribute', 'stemandleaf', 'quadrantal', 'independence', 'dependent', 'extreme','percentile', 'fundamental', 'polynomial', 'discrete', 'conjecture', 'quadrants', 'generalized', 'prediction', 'integrals', 'computing', 'trapezoidal', 'trials', 'affine', 'numerical', 'obtuse', 'index', 'inverse', 'hyperbolic','numerator', 'perpendicular', 'bisector', 'exponentiation', 'sine', 'plus', 'exponents', 'retangular', 'scatterplot', 'distance', 'enclosed', 'alpha', 'heptagon', 'median', 'intersection', 'comparing', 'shells', 'sinusoid', 'radius', 'counts', 'antiderivative', 'increase', 'proof', 'sum', 'enhancer', 'paradoxes', 'convergent', 'loglikelihood', 'matches', 'coordinates', 'derivative','nonparametric', 'doubling', 'strict', 'euclid', 'odds', 'trajectory', 'ordering', 'matrices', 'integral', 'factoring', 'polarrectangular', 'unified', 'cofunction', 'octahedron', 'cotangent', 'limiting', 'rectangle', 'composite', 'decreasing', 'reflecting', 'single', 'distributions', 'finite', 'circle', 'antidifferentation', 'inequalities', 'condensed', 'transform', 'modified', 'analytic', 'geometric', 'graphical', 'multiplicity', 'intensity', 'parameterization', 'dynamic', 'rules', 'contradiction', 'modeling', 'estimates', 'composition', 'parallelogram', 'slope', 'accuracy', 'databases', 'interquartile', 'alternate', 'versus', 'alternating', 'infinitesimal', 'inclusive', 'scatter', 'completeness', 'colin', 'theorem', 'rate', 'modular', 'comparison', 'diagonal', 'complementary', 'parallel', 'ellipsoid', 'criterion', 'producer', 'cosine', 'cos', 'quadruple', 'cubic', 'van', 'contingency', 'inscribed', 'selection', 'coefficient', 'calculating', 'metric', 'beta', 'hyperbola', 'oblate', 'concavity', 'periodicity', 'counterexample', 'observed', 'bisect', 'units', 'reduction',  'quadratic', 'ndimensional', 'substitution', 'descent', 'decrease', 'circular', 'simplify', 'laplace', 'marquis', 'elimination', 'integers', 'probabilities', 'information',  'covariance', 'form', 'length', 'frequency', 'truncating',  'collinear', 'parabolic', 'lateral', 'diametrically', 'subadditivity', 'min', 'analysis', 'rule', 'domains', 'linear', 'imaginary', 'set', 'compression', 'substitutions', 'data', 'optimization', 'proportional', 'fractal', 'scheme', 'congruent', 'differences', 'critical', 'range', 'invertible', 'quadrilateral', 'bound', 'axes', 'main', 'sums', 'upper', 'consolidation', 'halfangle', 'figure', 'stretch', 'paraxial', 'processes', 'pvalues', 'unordered', 'interpolation', 'platonic', 'opposed', 'cartesian', 'order', 'secant', 'polyploid', 'condition', 'time', 'variances', 'graphs', 'partial', 'numbers', 'probability', 'crosssections', 'parametric', 'uncountable', 'promoter', 'scores', 'recognizing', 'subpopulations', 'similar', 'contest', 'bounded', 'cardinality', 'minute', 'rates', 'below', 'disks', 'problem', 'cosecant', 'restricted', 'union', 'formulas', 'frequencies', 'fisheryates', 'loss', 'standard', 'convex', 'ramp', 'strophoid', 'higher', 'cochleoid', 'intuitive', 'scalar', 'mixing', 'annum', 'opposite', 'find', 'projectile', 'initialvalue', 'connected', 'circumference', 'triple', 'triangle', 'zones', 'indicator', 'solution', 'center', 'line', 'supply', 'sin', 'ordinal', 'quintic', 'vertices', 'partitions', 'iff', 'nonreal', 'definite', 'roberval', 'halls', 'variation', 'nondifferentiable', 'circumcenter', 'solving', 'steepest', 'forward', 'representation', 'classical', 'total', 'aperiodicity', 'table', 'irreducibility', 'epicycloids', 'solid', 'looking', 'vector','ellipse', 'breaks', 'convention', 'rotation', 'dot', 'affectedsonly', 'reflexive', 'uncountably', 'surd', 'lines', 'identities','exponent', 'property', 'contraction', 'equation', 'squares', 'unstable', 'ordered','definiteness', 'difference', 'cissoid', 'orders', 'epsilon', 'notation', 'step', 'endpoint', 'delayed', 'pointslope','measurement', 'priority', 'capacity', 'linearization', 'odd', 'sets', 'multiple', 'computation', 'drift', 'variable', 'domain', 'biconditional', 'nonsingular', 'empty', 'properties', 'sphere', 'divergent', 'restriction', 'borelcantelli', 'permutation', 'converge', 'constant', 'remainder', 'argument', 'trinomial', 'simultaneous', 'absolute', 'current', 'factor', 'solve', 'surfaces', 'scalene', 'trigonometric','application', 'discontinuous', 'consecutive', 'lagranges', 'summation', 'statistic', 'root', 'graphing', 'cost', 'expected', 'equilateral', 'modulation', 'triangles', 'sketching', 'quadrant', 'invariant', 'charge', 'norm', 'hydrostatic', 'outlier', 'curvature', 'efficiency', 'manipulation', 'system','expression', 'implicit', 'average', 'indirect', 'testing', 'polynomials', 'epoch', 'interphase', 'octants', 'panel', 'limits', 'overlap', 'central', 'something', 'potential', 'sample', 'covering', 'mixed', 'compounded', 'negatively', 'secondorder', 'absolutely', 'epitrochoid', 'mapping', 'pattern', 'herons', 'definition', 'value', 'binomial', 'approximate', 'models', 'hexahedron', 'surgery', 'minimum', 'noninvertible', 'path', 'eulers', 'recessive', 'modulo', 'cross', 'intercepts', 'propensity', 'significance', 'multilevel', 'tangents', 'elusive', 'closed', 'marginal', 'rainbows', 'period', 'movie', 'bivariate', 'differentials', 'angle', 'procedure', 'coefficients', 'haptoglobin', 'regulatory', 'curve', 'asymptotes', 'boxplot', 'infinite', 'modulus', 'significant', 'number', 'nonoverlapping','transpose', 'positively', 'reciprocal','gradient', 'differential', 'pyramid', 'combinatorics', 'multivariate', 'conversion', 'adjugate', 'minus', 'investment', 'fields', 'multivariable', 'founder', 'sister', 'reduced', 'optics', 'continuous', 'reduce', 'discriminant', 'max', 'exterior', 'hypothesis', 'positive', 'operations', 'input', 'additive', 'scale', 'transitions', 'digit', 'polyhedra', 'intermediate', 'adjacent', 'diameter', 'rounding', 'congruence', 'combination', 'pendulum', 'shifted', 'nonnegative', 'multiplication', 'product', 'process', 'output', 'venn', 'tables', 'coaster', 'sites', 'vertical', '', 'dimension', 'introducing', 'descartes', 'distribution', 'infinity', 'pitfalls', 'zone', 'gas', 'series', 'velocity', 'translation', 'horizontal', 'intercept', 'disjunction', 'approximations', 'common', 'modus', 'determinant'}
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
    final_list=list(i for i in clean_list if i in corpusraw)
    final_strings = ' '.join(final_list)
    return final_strings

import os
files1 = os.listdir('src\cityu')
files2 = os.listdir('src\mcgill')
files3= os.listdir('src\edin')
# Define your documents as a list of strings
documentscityu = [normalizer(pdf_reader(PyPDF2.PdfReader(os.path.join('src\cityu',file)))) for file in files1]
documentsmcgill = [normalizer(pdf_reader(PyPDF2.PdfReader(os.path.join('src\mcgill',file)))) for file in files2]
documentsedin = [normalizer(pdf_reader(PyPDF2.PdfReader(os.path.join('src\edin',file)))) for file in files3]
documents = documentscityu + documentsmcgill + documentsedin
# Convert the documents to a TF-IDF matrix
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(documents)

# Train a linear SVM classifier on the TF-IDF matrix
classifier = LinearSVC()
classifier.fit(tfidf_matrix, [i for i in range(len(documents))])

# Use the trained classifier to predict the similarity between the documents
similarity = classifier.predict(tfidf_matrix)
print(similarity)

'''
SVM results shows that no two documents are the same
this means that the corpus is not strong enough to proof
[  0   1   2   3   4   5   6   7   8  11  10  11  12  13  14  15  16  17
  18  19  20  21  22  23  24  25  26  27  28  29  30  31  32  33  34  35
  36  37  38  39  40  41  42  43  44  45  46  47  48  49  50  51  52  53
  54  55  56  57  58  59  60  61  62  63  64  65  66  67  68  69  70  71
  72  73  74  75  76  77  78  79  80  81  82  83  84  85  86  87  88  89
  90  91  92  93  94  95  96  97  98  99 100 101 102 103 104 105 106 107
 108 109 110 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125
 126 127 128 129 130 131 132 133 134 135 136 137 138 139 140 141 142 143
 144 145 146 147 148 149 150 151 152 153 154 155 156 157 158 159 160 161
 162 163 164 165 166 167 168 169 170 171 172 173 174 175 176 177 178 179
 180 181 182 183 184 185 186 187 188 189 190 191 192]
'''
