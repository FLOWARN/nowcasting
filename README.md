# Installation instructions of nowcasting library


The following are the steps to install the required libraries to run the Jupyter notebooks for the tasks such as data download, nowcasting, visualization and evaluation. These instructions are currently only intended for implementation on Linux based systems.


1.	Clone the GitHub repository from the following link: https://github.com/FLOWARN/nowcasting.git 
2. Change directory

```
cd nowcasting/
```

3.	Create conda environment from the tito_env.yml file.
```
conda env create -f tito_env.yml   
```

4. Activate the conda environment
```
conda activate tito_env
```
5.	Install the servir package locally
```
pip install -e .
```

6.	Change into the servir_nowcasting_examples/ directory
```
cd servir_nowcasting_examples/
```
7.	Create `temp` and `data`  folder as it is present in the google drive: https://drive.google.com/drive/folders/1dWK8wDNKB3XwRuW6mNX22KHI1hyPH6uQ?usp=sharing 

8.	Create an empty directory called `results`.


After following these steps, the user should be in a position to run all the notebooks present in `servir_nowcasting_examples`
