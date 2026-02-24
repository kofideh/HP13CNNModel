The simple way to run the code is in Google Colab.
But if you want to run it locally, install the required packages using the following command:
pip install numpy scipy torch nibabel matplotlib scikit-learn pandas.

To train the model, run the following command:
python train_hybrid_multihead.py  

after specifying acquisition parameters in the code.

To test the model, run the following command:
python robust_clinical_inference.py 

after specifying the weight path in the code.