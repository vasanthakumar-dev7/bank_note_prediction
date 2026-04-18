<div align="center">

#  Bank Note Authentication using Machine Learning
### *Detecting genuine vs fake banknotes with Random Forest*

![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--Learn-ML-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![Model](https://img.shields.io/badge/Model-Random%20Forest-2E8B57?style=for-the-badge)
![Accuracy](https://img.shields.io/badge/Accuracy-97%25-brightgreen?style=for-the-badge)

> *A simple yet effective ML model to classify banknotes as real or fake based on statistical features.*

</div>



##  Overview

This project implements a **Machine Learning model** to classify whether a banknote is **authentic or forged** using statistical features extracted from images.

The model is built using a **Random Forest Classifier**, known for its robustness and high accuracy in classification tasks.



## 🧠 Features Used

The model takes the following input features:


 `variance` 
 `skewness`
 `curtosis` 
 `entropy` 



##  Model Details

- **Algorithm:** Random Forest Classifier  
- **Type:** Supervised Learning 
- **Library:** Scikit-learn  
- **Dataset:** Banknote Authentication Dataset  



## 📊 Performance

- ✅ **Accuracy:** ~97%  
-  High precision and recall across classes  
-  Handles non-linear relationships effectively  

## How to Run
git clone https://github.com/vasanthakumar-dev7/bank_note_prediction

cd bank_note_prediction

python model.py


##  Conclusion


This project demonstrates how simple statistical features combined with a Random Forest model can effectively detect counterfeit banknotes with high accuracy.
