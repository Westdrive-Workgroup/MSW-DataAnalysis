* Encoding: UTF-8.
MANOVA Intention Usefulness Ease Trust BY AgeGroup(0,4) Gender(0,1) Condition(0,2) 
/DISCRIM=STAN RAW CORR
/PRINT=SIGNIF(MULTIV,UNIV,EIGEN,DIMENR)
/DESIGN AgeGroup, Gender, Condition, AgeGroup by Gender


COMPUTE Super_Condition = (-.03420 * Intention) + (.03247 * Usefulness)
+ (-.01176 * Ease) + (.01498 * Trust).
EXECUTE.

COMPUTE Super_Gender = (-.01026 * Intention) + (-.00212 * Usefulness)
+ (-.02059 * Ease) + (-.00629 * Trust).
EXECUTE.

COMPUTE Super_AgeGroup = (-.00850 * Intention) + (-.00647 * Usefulness)
+ (-.01570 * Ease) + (-.01029 * Trust).
EXECUTE.

COMPUTE Super_AgeGroup_2 = (.01947 * Intention) + (.01436 * Usefulness)
+ (-.00371 * Ease) + (-.03451 * Trust).
EXECUTE.

COMPUTE Super_AgeGroup_Gender = (-.02251 * Intention) + (.00903 * Usefulness)
+ (-.02492 * Ease) + (-.01862 * Trust).
EXECUTE.
