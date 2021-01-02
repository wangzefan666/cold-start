# Heater--Cold-Start-Recommendation
Code for the SIGIR20 paper -- Recommendation for New Users and New Items via Randomized Training and Mixture-of-Experts Transformation

Fix some bugs in the origin code.  

## Data
Used three datasets (CiteULike, LastFM, and XING) in this work, which are stored in 'data' folder. Details about these datasets can be found in the paper and also the original papers cited in our paper.

## Requirements
- python 3  
- tensorflow-gpu 1.5.0 


## Excution
- Run `python main_CiteULike.py` to run the model Heater on CiteULike data; 
- Run `python main_LastFM.py` to run the model Heater on LastFM data; 
- Run `python main_XING.py` to run the model Heater on XING data;

## Thanks
Code is based on the implementation of DropoutNet (https://github.com/layer6ai-labs/DropoutNet).

