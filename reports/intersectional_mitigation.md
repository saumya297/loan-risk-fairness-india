
INTERSECTIONAL FAIRNESS MITIGATION SUMMARY
==========================================
Method: ExponentiatedGradient with EqualizedOdds
Sensitive feature: Gender × Property Area (intersectional)

BEFORE mitigation:
  Accuracy: 0.7967
  DP Difference (intersectional): 0.5357

AFTER mitigation:
  Accuracy: 0.8455
  DP Difference (intersectional): 0.5325
  Fairness improvement: +0.6%

Group-level changes:
                     before  after  change
sensitive_feature_0                       
Female | Rural         25.0   50.0    25.0
Female | Semiurban     78.6   71.4    -7.2
Female | Urban         28.6   28.6     0.0
Male | Rural           63.3   63.3     0.0
Male | Semiurban       77.1   80.0     2.9
Male | Urban           72.7   81.8     9.1
