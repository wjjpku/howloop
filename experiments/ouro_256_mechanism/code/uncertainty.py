import math
from scipy.stats import beta,binomtest

def wilson(k,n):
 if not n:return None
 z=1.959963984540054;p=k/n;den=1+z*z/n;mid=(p+z*z/(2*n))/den;half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
 return [max(0.,mid-half),min(1.,mid+half)]

def paired_exact(a_only,b_only,n):
 def interval(k):return [float(beta.ppf(.0125,k,n-k+1)) if k else 0.,float(beta.ppf(.9875,k+1,n-k)) if k<n else 1.]
 aa,bb=interval(a_only),interval(b_only)
 return {'conservative_ci95':[aa[0]-bb[1],aa[1]-bb[0]],'exact_p_two_sided':float(binomtest(a_only,a_only+b_only,.5).pvalue) if a_only+b_only else 1.,'interval_method':'Bonferroni simultaneous97.5% Clopper-Pearson intervals for discordant proportions'}
