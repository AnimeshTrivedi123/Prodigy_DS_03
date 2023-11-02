{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {
    "id": "XvUb9a07X3XK",
    "tags": []
   },
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "from sklearn.cluster import KMeans\n",
    "from sklearn import datasets\n",
    "from io import StringIO\n",
    "from sklearn.tree import export_graphviz\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn import tree\n",
    "from sklearn import metrics\n",
    "%matplotlib inline"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 226
    },
    "id": "kO6zzeOCacF0",
    "outputId": "7f634311-49a3-447d-c365-c0ed54e45e09",
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>age</th>\n",
       "      <th>job</th>\n",
       "      <th>marital</th>\n",
       "      <th>education</th>\n",
       "      <th>default</th>\n",
       "      <th>balance</th>\n",
       "      <th>housing</th>\n",
       "      <th>loan</th>\n",
       "      <th>contact</th>\n",
       "      <th>day</th>\n",
       "      <th>month</th>\n",
       "      <th>duration</th>\n",
       "      <th>campaign</th>\n",
       "      <th>pdays</th>\n",
       "      <th>previous</th>\n",
       "      <th>poutcome</th>\n",
       "      <th>deposit</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>59</td>\n",
       "      <td>admin.</td>\n",
       "      <td>married</td>\n",
       "      <td>secondary</td>\n",
       "      <td>no</td>\n",
       "      <td>2343</td>\n",
       "      <td>yes</td>\n",
       "      <td>no</td>\n",
       "      <td>unknown</td>\n",
       "      <td>5</td>\n",
       "      <td>may</td>\n",
       "      <td>1042</td>\n",
       "      <td>1</td>\n",
       "      <td>-1</td>\n",
       "      <td>0</td>\n",
       "      <td>unknown</td>\n",
       "      <td>yes</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>56</td>\n",
       "      <td>admin.</td>\n",
       "      <td>married</td>\n",
       "      <td>secondary</td>\n",
       "      <td>no</td>\n",
       "      <td>45</td>\n",
       "      <td>no</td>\n",
       "      <td>no</td>\n",
       "      <td>unknown</td>\n",
       "      <td>5</td>\n",
       "      <td>may</td>\n",
       "      <td>1467</td>\n",
       "      <td>1</td>\n",
       "      <td>-1</td>\n",
       "      <td>0</td>\n",
       "      <td>unknown</td>\n",
       "      <td>yes</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>41</td>\n",
       "      <td>technician</td>\n",
       "      <td>married</td>\n",
       "      <td>secondary</td>\n",
       "      <td>no</td>\n",
       "      <td>1270</td>\n",
       "      <td>yes</td>\n",
       "      <td>no</td>\n",
       "      <td>unknown</td>\n",
       "      <td>5</td>\n",
       "      <td>may</td>\n",
       "      <td>1389</td>\n",
       "      <td>1</td>\n",
       "      <td>-1</td>\n",
       "      <td>0</td>\n",
       "      <td>unknown</td>\n",
       "      <td>yes</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>55</td>\n",
       "      <td>services</td>\n",
       "      <td>married</td>\n",
       "      <td>secondary</td>\n",
       "      <td>no</td>\n",
       "      <td>2476</td>\n",
       "      <td>yes</td>\n",
       "      <td>no</td>\n",
       "      <td>unknown</td>\n",
       "      <td>5</td>\n",
       "      <td>may</td>\n",
       "      <td>579</td>\n",
       "      <td>1</td>\n",
       "      <td>-1</td>\n",
       "      <td>0</td>\n",
       "      <td>unknown</td>\n",
       "      <td>yes</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>54</td>\n",
       "      <td>admin.</td>\n",
       "      <td>married</td>\n",
       "      <td>tertiary</td>\n",
       "      <td>no</td>\n",
       "      <td>184</td>\n",
       "      <td>no</td>\n",
       "      <td>no</td>\n",
       "      <td>unknown</td>\n",
       "      <td>5</td>\n",
       "      <td>may</td>\n",
       "      <td>673</td>\n",
       "      <td>2</td>\n",
       "      <td>-1</td>\n",
       "      <td>0</td>\n",
       "      <td>unknown</td>\n",
       "      <td>yes</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   age         job  marital  education default  balance housing loan  contact  \\\n",
       "0   59      admin.  married  secondary      no     2343     yes   no  unknown   \n",
       "1   56      admin.  married  secondary      no       45      no   no  unknown   \n",
       "2   41  technician  married  secondary      no     1270     yes   no  unknown   \n",
       "3   55    services  married  secondary      no     2476     yes   no  unknown   \n",
       "4   54      admin.  married   tertiary      no      184      no   no  unknown   \n",
       "\n",
       "   day month  duration  campaign  pdays  previous poutcome deposit  \n",
       "0    5   may      1042         1     -1         0  unknown     yes  \n",
       "1    5   may      1467         1     -1         0  unknown     yes  \n",
       "2    5   may      1389         1     -1         0  unknown     yes  \n",
       "3    5   may       579         1     -1         0  unknown     yes  \n",
       "4    5   may       673         2     -1         0  unknown     yes  "
      ]
     },
     "execution_count": 2,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Load data file\n",
    "bank=pd.read_csv(\"C:\\\\Users\\\\LENOVO\\\\Downloads\\\\bank.csv\")\n",
    "bank.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "7-OiKgQjawIP",
    "outputId": "617ba522-71b9-4cd7-bfdf-4a8e09f01353",
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "age          0\n",
       "job          0\n",
       "marital      0\n",
       "education    0\n",
       "default      0\n",
       "balance      0\n",
       "housing      0\n",
       "loan         0\n",
       "contact      0\n",
       "day          0\n",
       "month        0\n",
       "duration     0\n",
       "campaign     0\n",
       "pdays        0\n",
       "previous     0\n",
       "poutcome     0\n",
       "deposit      0\n",
       "dtype: int64"
      ]
     },
     "execution_count": 3,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Check if the data set contains any null values - Nothing found!\n",
    "bank[bank.isnull().any(axis=1)].count()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 300
    },
    "id": "5XViSq1UazSP",
    "outputId": "504d0c7a-144c-4b90-d8f1-ace46d6cdec1",
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>age</th>\n",
       "      <th>balance</th>\n",
       "      <th>day</th>\n",
       "      <th>duration</th>\n",
       "      <th>campaign</th>\n",
       "      <th>pdays</th>\n",
       "      <th>previous</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>count</th>\n",
       "      <td>11162.000000</td>\n",
       "      <td>11162.000000</td>\n",
       "      <td>11162.000000</td>\n",
       "      <td>11162.000000</td>\n",
       "      <td>11162.000000</td>\n",
       "      <td>11162.000000</td>\n",
       "      <td>11162.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>mean</th>\n",
       "      <td>41.231948</td>\n",
       "      <td>1528.538524</td>\n",
       "      <td>15.658036</td>\n",
       "      <td>371.993818</td>\n",
       "      <td>2.508421</td>\n",
       "      <td>51.330407</td>\n",
       "      <td>0.832557</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>std</th>\n",
       "      <td>11.913369</td>\n",
       "      <td>3225.413326</td>\n",
       "      <td>8.420740</td>\n",
       "      <td>347.128386</td>\n",
       "      <td>2.722077</td>\n",
       "      <td>108.758282</td>\n",
       "      <td>2.292007</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>min</th>\n",
       "      <td>18.000000</td>\n",
       "      <td>-6847.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>2.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>-1.000000</td>\n",
       "      <td>0.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>25%</th>\n",
       "      <td>32.000000</td>\n",
       "      <td>122.000000</td>\n",
       "      <td>8.000000</td>\n",
       "      <td>138.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>-1.000000</td>\n",
       "      <td>0.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>50%</th>\n",
       "      <td>39.000000</td>\n",
       "      <td>550.000000</td>\n",
       "      <td>15.000000</td>\n",
       "      <td>255.000000</td>\n",
       "      <td>2.000000</td>\n",
       "      <td>-1.000000</td>\n",
       "      <td>0.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>75%</th>\n",
       "      <td>49.000000</td>\n",
       "      <td>1708.000000</td>\n",
       "      <td>22.000000</td>\n",
       "      <td>496.000000</td>\n",
       "      <td>3.000000</td>\n",
       "      <td>20.750000</td>\n",
       "      <td>1.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>max</th>\n",
       "      <td>95.000000</td>\n",
       "      <td>81204.000000</td>\n",
       "      <td>31.000000</td>\n",
       "      <td>3881.000000</td>\n",
       "      <td>63.000000</td>\n",
       "      <td>854.000000</td>\n",
       "      <td>58.000000</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                age       balance           day      duration      campaign  \\\n",
       "count  11162.000000  11162.000000  11162.000000  11162.000000  11162.000000   \n",
       "mean      41.231948   1528.538524     15.658036    371.993818      2.508421   \n",
       "std       11.913369   3225.413326      8.420740    347.128386      2.722077   \n",
       "min       18.000000  -6847.000000      1.000000      2.000000      1.000000   \n",
       "25%       32.000000    122.000000      8.000000    138.000000      1.000000   \n",
       "50%       39.000000    550.000000     15.000000    255.000000      2.000000   \n",
       "75%       49.000000   1708.000000     22.000000    496.000000      3.000000   \n",
       "max       95.000000  81204.000000     31.000000   3881.000000     63.000000   \n",
       "\n",
       "              pdays      previous  \n",
       "count  11162.000000  11162.000000  \n",
       "mean      51.330407      0.832557  \n",
       "std      108.758282      2.292007  \n",
       "min       -1.000000      0.000000  \n",
       "25%       -1.000000      0.000000  \n",
       "50%       -1.000000      0.000000  \n",
       "75%       20.750000      1.000000  \n",
       "max      854.000000     58.000000  "
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "bank.describe()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 449
    },
    "id": "fx-MTksOa0gn",
    "outputId": "e963fed1-b896-4c9a-b8df-34324e161a7c",
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAggAAAGwCAYAAADMjZ3mAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjcuMiwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8pXeV/AAAACXBIWXMAAA9hAAAPYQGoP6dpAAAbnklEQVR4nO3dfZCV5Xn48essK7tL2cWEVGUVFKwtNSBISKzWKtMwGkSmiZ2ogC8p0zRmNEWpkSZayERTbOw4TfuHJsZxasTodHwZYwqoieJQp/hCUbSWkGCLVgmtCbKGAmH3/v2R2fPjcO0qEODA2c9nZmd47ufh7H2xsOfLeYFKKaUEAMAumuq9AQDg0CMQAIBEIAAAiUAAABKBAAAkAgEASAQCAJA07+tP7OnpiTfffDPa29ujUqnszz0BAAdIKSW6urqis7Mzmpr6f5xgnwPhzTffjJEjR+7rTwcA6uj111+P4447rt/z+xwI7e3t1U/Q0dGxrzcDABxEW7ZsiZEjR1bvx/uzz4HQ+7RCR0eHQACAw8z7vTzAixQBgEQgAACJQAAAEoEAACQCAQBIBAIAkAgEACARCABAIhAAgEQgAACJQAAAEoEAACQCAQBIBAIAkAgEACARCABAIhAAgEQgAACJQAAAEoEAACQCAQBIBAIAkAgEACARCABAIhAAgEQgAACJQAAAEoEAACQCAQBIBAIAkAgEACARCABA0lzvDbD/lVJi27Zt9d7GeyqlxPbt2yMioqWlJSqVSp13dHC0trYOmFmBw5tAaEDbtm2LadOm1Xsb9GHJkiXR1tZW720AvC9PMQAAiUcQGty7E2dGaToEv8zdv4z2F++LiIiuCRdHDDqizhs6cCo9O2Po6u/WexsAe+UQvOdgfypNzYf+ne+gIw79Pf4aSr03ALAPPMUAACQCAQBIBAIAkAgEACARCABAIhAAgEQgAACJQAAAEoEAACQCAQBIBAIAkAgEACARCABAIhAAgEQgAACJQAAAEoEAACQCAQBIBAIAkAgEACARCABAIhAAgEQgAACJQAAAEoEAACQCAQBIBAIAkAgEACARCABAIhAAgEQgAACJQAAAEoEAACQCAQBIBAIAkAgEACARCABAIhAAgEQgAACJQAAAEoEAACQCAQBIBAIAkAgEACARCABAIhAAgEQgAACJQAAAEoEAACQCAQBIBAIAkAgEACARCABAIhAAgEQgAACJQAAAEoEAACQCAQBIBAIAkAgEACARCABAIhAAgEQgAACJQAAAEoEAACQCAQBIBAIAkAgEACARCABA0lzvDeyqlBLbtm2LiIjW1taoVCp13hHAocH3Rw62Q+oRhG3btsW0adNi2rRp1T8IAPj+yMF3SAUCAHBoEAgAQCIQAIBEIAAAiUAAABKBAAAkAgEASAQCAJAIBAAgEQgAQCIQAIBEIAAAiUAAABKBAAAkAgEASAQCAJAIBAAgEQgAQCIQAIBEIAAAiUAAABKBAAAkAgEASAQCAJAIBAAgEQgAQCIQAIBEIAAAiUAAABKBAAAkAgEASAQCAJAIBAAgEQgAQCIQAIBEIAAAiUAAABKBAAAkAgEASAQCAJAIBAAgEQgAQCIQAIBEIAAAiUAAABKBAAAkAgEASAQCAJAIBAAgEQgAQCIQAIBEIAAAiUAAABKBAAAkAgEASAQCAJAIBAAgEQgAQCIQAIBEIAAAiUAAABKBAAAkAgEASAQCAJAIBAAgEQgAQCIQAIBEIAAAiUAAABKBAHCYmzJlSvXjQKxPnTo1pkyZElOnTn3fa88555yYMmVKnHPOOe97GxERM2bMiClTpsSMGTNq1i+77LKYMmVKXHbZZfu8ftVVV8WUKVPiqquuqrn2zjvvjD/8wz+MO++8s2b9mWeeiYsuuiieeeaZPbq+P/1dvze3099eDiaBAHAYu+222/o8vuWWW2rWe4/nzp1bs957/A//8A81673HTzzxROzcuTMiInbu3BlPPPFE3HHHHTXX9h4//fTTsWPHjoiI2LFjRzz99NP93kZExKpVq6KrqysiIrq6umLVqlUREbFu3brYsGFDRERs2LAh1q1bt9frGzZsiJdffjkiIl5++eXq+c2bN8fixYujp6cnFi9eHJs3b46IiG3btsWtt94aP/3pT+PWW2+Nbdu2vef1/env+r25nf72crAJBIDD2P3339/n8fe///2a9d7jF198sWa99/iBBx6oWe89vummm2rWb7rppli8eHHNWu/xggULatZ7j/u6jYiIefPm1az3Hn/+85+vWe893pv1K664omat9/iv/uqvoqenJyIienp6qntcvHhxvP322xER8fbbb8e99977ntf3p7/r9+Z2+tvLwdZcl8/aj1JK9cf1KqZGUPNrt8uvKXXi9zX7wa6/d3q/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nD/72c/K8ccfXz7zmc+UlStXltdee6088cQT5cc//nH1mkaZddOmTTVfz8cff7xERHnyySdLKY0zZyml3HTTTWX48OHl0UcfLa+99lr5p3/6pzJ06NDyd3/3d9VrGmXeCy+8sJx88sll+fLlZd26dWXhwoWlo6OjvPHGG6WUxplzTx0WgbC7TZs2lYgoy5cvL6WU0tPTU4455phy8803V6/Ztm1bGTZsWLn99tvrtc395gMf+ED59re/3ZBzdnV1lZNOOqk8/vjj5eyzz64GQiPNunDhwjJhwoQ+zzXSnPPnzy9nnnlmv+cbadbdzZ07t5x44omlp6en4eacPn16mTNnTs3aBRdcUC655JJSSuN8Xbdu3VoGDRpUHn300Zr1CRMmlOuvv75h5twbh8VTDLt75513IiLigx/8YEREvPbaa7Fx48Y455xzqte0tLTE2WefHc8880xd9rg/dHd3x3333Re/+MUv4vTTT2/IOa+88sqYPn16TJ06tWa90WZdt25ddHZ2xujRo+Piiy+O9evXR0RjzfnII4/E5MmT49Of/nQcddRRceqpp8Ydd9xRPd9Is+5qx44dcc8998ScOXOiUqk03Jxnnnlm/OAHP4gf/ehHERHx4osvxooVK+K8886LiMb5uu7cuTO6u7ujtbW1Zr2trS1WrFjRMHPujcMuEEopMW/evDjzzDNj3LhxERGxcePGiIg4+uija649+uijq+cOJ2vWrImhQ4dGS0tLXHHFFfHQQw/FySef3HBz3nfffbFq1apYtGhROtdIs5522mlx9913x7Jly+KOO+6IjRs3xhlnnBFvv/12Q825fv36uO222+Kkk06KZcuWxRVXXBF//ud/HnfffXdENNbXdFcPP/xwbN68OT7zmc9EROPNOX/+/Jg5c2aMHTs2jjjiiDj11FPj6quvjpkzZ0ZE48zb3t4ep59+etx4443x5ptvRnd3d9xzzz2xcuXKeOuttxpmzr2xz/+bY71cddVV8dJLL8WKFSvSud3/2+lSSl3/K+p99Tu/8zuxevXq2Lx5czzwwANx+eWXx/Lly6vnG2HO119/PebOnRuPPfZYKvZdNcKs06ZNq/54/Pjxcfrpp8eJJ54Y//iP/xi/93u/FxGNMWdPT09Mnjw5/vqv/zoiIk499dR45ZVX4rbbbovLLrusel0jzLqrO++8M6ZNmxadnZ01640y5/333x/33HNP3HvvvfHhD384Vq9eHVdffXV0dnbG5ZdfXr2uEeb9zne+E3PmzIljjz02Bg0aFJMmTYpZs2bFqlWrqtc0wpx76rB6BOELX/hCPPLII/Hkk0/W/FfTva8I373iNm3alGrvcDB48OD4rd/6rZg8eXIsWrQoJkyYEN/4xjcaas4XXnghNm3aFB/5yEeiubk5mpubY/ny5fH3f//30dzcXJ2nEWbd3W/8xm/E+PHjY926dQ31NR0xYkScfPLJNWu/+7u/Gxs2bIiIxvtzGhHxX//1X/HEE0/En/7pn1bXGm3OL37xi/GXf/mXcfHFF8f48ePj0ksvjWuuuab6yF8jzXviiSfG8uXL4913343XX389nn322fjlL38Zo0ePbqg599RhEQillLjqqqviwQcfjB/+8IcxevTomvO9X7zHH3+8urZjx45Yvnx5nHHGGQd7u/tdKSW2b9/eUHN+/OMfjzVr1sTq1aurH5MnT47Zs2fH6tWrY8yYMQ0z6+62b98er776aowYMaKhvqa///u/n95+/KMf/SiOP/74iGjMP6d33XVXHHXUUTF9+vTqWqPNuXXr1mhqqr2rGDRoUPVtjo02b8SvIn7EiBHx85//PJYtWxZ/9Ed/1JBzvq86vThyr3z+858vw4YNK0899VTNW4u2bt1avebmm28uw4YNKw8++GBZs2ZNmTlz5mH59pMvfelL5emnny6vvfZaeemll8qXv/zl0tTUVB577LFSSuPM2Zdd38VQSuPM+hd/8RflqaeeKuvXry//+q//Ws4///zS3t5e/vM//7OU0jhzPvvss6W5ubl87WtfK+vWrSuLFy8uQ4YMKffcc0/1mkaZtZRSuru7y6hRo8r8+fPTuUaa8/LLLy/HHnts9W2ODz74YPnQhz5Urrvuuuo1jTLv0qVLy5IlS8r69evLY489ViZMmFA+9rGPlR07dpRSGmfOPXVYBEJE9Plx1113Va/p6ekpCxcuLMccc0xpaWkpZ511VlmzZk39Nr2P5syZU44//vgyePDg8pu/+Zvl4x//eDUOSmmcOfuyeyA0yqy975U+4ogjSmdnZ7ngggvKK6+8Uj3fKHOWUsr3vve9Mm7cuNLS0lLGjh1bvvWtb9Wcb6RZly1bViKirF27Np1rpDm3bNlS5s6dW0aNGlVaW1vLmDFjyvXXX1+2b99evaZR5r3//vvLmDFjyuDBg8sxxxxTrrzyyrJ58+bq+UaZc0/5754BgOSweA0CAHBwCQQAIBEIAEAiEACARCAAAIlAAAASgQAAJAIBAEgEAgCQCAQAIBEIAEAiEGAAWbp0aZx55plx5JFHxvDhw+P888+Pn/zkJ9XzzzzzTEycODFaW1tj8uTJ8fDDD0elUonVq1dXr/n3f//3OO+882Lo0KFx9NFHx6WXXhr/+7//W4dpgANJIMAA8otf/CLmzZsXzz33XPzgBz+Ipqam+NSnPhU9PT3R1dUVM2bMiPHjx8eqVavixhtvjPnz59f8/LfeeivOPvvsmDhxYjz//POxdOnS+OlPfxoXXnhhnSYCDhT/myMMYP/zP/8TRx11VKxZsyZWrFgRN9xwQ7zxxhvR2toaERHf/va347Of/Wz827/9W0ycODEWLFgQK1eujGXLllVv44033oiRI0fG2rVr47d/+7frNQqwn3kEAQaQn/zkJzFr1qwYM2ZMdHR0xOjRoyMiYsOGDbF27do45ZRTqnEQEfGxj32s5ue/8MIL8eSTT8bQoUOrH2PHjq3eNtA4muu9AeDgmTFjRowcOTLuuOOO6OzsjJ6enhg3blzs2LEjSilRqVRqrt/9Acaenp6YMWNG/M3f/E267REjRhzQvQMHl0CAAeLtt9+OV199Nb75zW/GH/zBH0RExIoVK6rnx44dG4sXL47t27dHS0tLREQ8//zzNbcxadKkeOCBB+KEE06I5mbfPqCReYoBBogPfOADMXz48PjWt74VP/7xj+OHP/xhzJs3r3p+1qxZ0dPTE3/2Z38Wr776aixbtiz+9m//NiKi+sjClVdeGT/72c9i5syZ8eyzz8b69evjscceizlz5kR3d3dd5gIODIEAA0RTU1Pcd9998cILL8S4cePimmuuiVtuuaV6vqOjI773ve/F6tWrY+LEiXH99dfHggULIiKqr0vo7OyMf/mXf4nu7u4499xzY9y4cTF37twYNmxYNDX5dgKNxLsYgH4tXrw4/uRP/iTeeeedaGtrq/d2gIPIk4hA1d133x1jxoyJY489Nl588cWYP39+XHjhheIABiCBAFRt3LgxFixYEBs3bowRI0bEpz/96fja175W720BdeApBgAg8aoiACARCABAIhAAgEQgAACJQAAAEoEAACQCAQBIBAIAkPw/gyfWnxwzc3UAAAAASUVORK5CYII=",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Boxplot for 'age'\n",
    "g = sns.boxplot(x=bank[\"age\"])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 466
    },
    "id": "2VU3cA2Ca6cF",
    "outputId": "abb76275-90ab-4468-a480-5c5b85c508a0",
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<Axes: xlabel='age', ylabel='Count'>"
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAjsAAAGwCAYAAABPSaTdAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjcuMiwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8pXeV/AAAACXBIWXMAAA9hAAAPYQGoP6dpAAAwGElEQVR4nO3df3TU1Z3/8ddAkuFXEgJIJsEAAaIWE4QGS0m1YPl1UFCXPaWKurhiVwsCEVgo4kK0lSitQA0Fi2UBRTaePYrYHwJBJZZmVRg3OklZivwQgglpMSSAcQLJ/f7hl0+dJAgJSWZy83yc8znHuZ87k/smmLy4n/v5XJcxxggAAMBS7YI9AAAAgOZE2AEAAFYj7AAAAKsRdgAAgNUIOwAAwGqEHQAAYDXCDgAAsFpYsAcQCmpqavTZZ58pMjJSLpcr2MMBAACXwRij06dPKz4+Xu3aXXz+hrAj6bPPPlNCQkKwhwEAABrh2LFjuvrqqy96nrAjKTIyUtJXf1hRUVFBHg0AALgcFRUVSkhIcH6PXwxhR3IuXUVFRRF2AABoZS61BIUFygAAwGqEHQAAYDXCDgAAsBphBwAAWI2wAwAArEbYAQAAViPsAAAAqxF2AACA1YIadjIyMuRyuQIOj8fjnDfGKCMjQ/Hx8erYsaNGjhypwsLCgM/w+/2aOXOmevTooc6dO+v2229XUVFRS5cCAABCVNBndq6//noVFxc7h8/nc84tW7ZMy5cv16pVq7Rnzx55PB6NGTNGp0+fdvqkp6dry5Ytys7O1u7du3XmzBlNmDBB1dXVwSgHAACEmKBvFxEWFhYwm3OBMUYrV67UokWLNGnSJEnSxo0bFRsbq82bN+uhhx5SeXm51q1bp5deekmjR4+WJG3atEkJCQnauXOnxo0b16K1AACA0BP0mZ0DBw4oPj5eiYmJuuuuu3To0CFJ0uHDh1VSUqKxY8c6fd1ut0aMGKG8vDxJktfr1blz5wL6xMfHKzk52elTH7/fr4qKioADAADYKahhZ9iwYXrxxRe1fft2vfDCCyopKVFaWppOnjypkpISSVJsbGzAe2JjY51zJSUlioiIUExMzEX71CczM1PR0dHOkZCQ0MSVAQCAUBHUsDN+/Hj98z//s1JSUjR69Gj94Q9/kPTV5aoLau9kaoy55O6ml+qzcOFClZeXO8exY8euoAoAABDKgr5m5+s6d+6slJQUHThwQHfeeaekr2Zv4uLinD6lpaXObI/H41FVVZXKysoCZndKS0uVlpZ20a/jdrvldrubpwg0iN/vl9frDWhLTU3l+wMAaDJBX7PzdX6/X/v27VNcXJwSExPl8XiUk5PjnK+qqlJubq4TZFJTUxUeHh7Qp7i4WAUFBd8YdhA6vF6vZq3eqsVbC7R4a4Fmrd5aJ/wAAHAlgjqzM2/ePE2cOFG9e/dWaWmpfv7zn6uiokJTp06Vy+VSenq6li5dqqSkJCUlJWnp0qXq1KmTpkyZIkmKjo7WtGnTNHfuXHXv3l3dunXTvHnznMtiaB269uqvHv1Tgj0MAIClghp2ioqKdPfdd+vvf/+7rrrqKn33u9/Ve++9pz59+kiS5s+fr8rKSk2fPl1lZWUaNmyYduzYocjISOczVqxYobCwME2ePFmVlZUaNWqUNmzYoPbt2werLAAAEEJcxhgT7EEEW0VFhaKjo1VeXq6oqKhgD6dNycvL0+KtBc7Mzt8P+vTkHclchgQAXNLl/v4OqTU7AAAATY2wAwAArEbYAQAAViPsAAAAqxF2AACA1Qg7AADAaoQdAABgNcIOAACwWkhtBArUh81CAQBXgrCDkHdhs9CuvfpLkk4dP6jnpounLAMALgthB60Cm4UCABqLNTsAAMBqhB0AAGA1wg4AALAaYQcAAFiNsAMAAKzG3ViwAs/iAQBcDGEHrU7N+XPy+XwBbT6fT2vfPaiYqwdI4lk8AIB/IOyg1ak4cVRZRyrlOeRy2ory/6SYAak8iwcAUAdhB61SpCcxINicOn4wiKMBAIQywg7aDNb1AEDbRNhBm8EeWwDQNhF20KawxxYAtD2EHbSY+i4j+Xw+1dQEaUAAgDaBsIMWU/sykvSPu6gAAGguhB20qNqXkbiLCgDQ3NguAgAAWI2wAwAArEbYAQAAViPsAAAAqxF2AACA1Qg7AADAaoQdAABgNcIOAACwGmEHAABYjbADAACsRtgBAABWI+wAAACrEXYAAIDVCDsAAMBqYcEeABAsNefPyefz1WlPTU2V2+0OwogAAM2BsIM2q+LEUWUdqZTnkMtpO3X8oJ6bLqWlpQVxZACApkTYQZsW6UlUj/4pwR4GAKAZsWYHAABYjbADAACsRtgBAABWI+wAAACrEXYAAIDVCDsAAMBqhB0AAGA1wg4AALAaYQcAAFiNsAMAAKxG2AEAAFYj7AAAAKsRdgAAgNUIOwAAwGqEHQAAYDXCDgAAsBphBwAAWC1kwk5mZqZcLpfS09OdNmOMMjIyFB8fr44dO2rkyJEqLCwMeJ/f79fMmTPVo0cPde7cWbfffruKiopaePQAACBUhUTY2bNnj9auXatBgwYFtC9btkzLly/XqlWrtGfPHnk8Ho0ZM0anT592+qSnp2vLli3Kzs7W7t27debMGU2YMEHV1dUtXQYAAAhBQQ87Z86c0T333KMXXnhBMTExTrsxRitXrtSiRYs0adIkJScna+PGjfriiy+0efNmSVJ5ebnWrVunZ599VqNHj9aQIUO0adMm+Xw+7dy5M1glAQCAEBL0sDNjxgzddtttGj16dED74cOHVVJSorFjxzptbrdbI0aMUF5eniTJ6/Xq3LlzAX3i4+OVnJzs9KmP3+9XRUVFwAEAAOwUFswvnp2drQ8//FB79uypc66kpESSFBsbG9AeGxurTz/91OkTERERMCN0oc+F99cnMzNTTzzxxJUOHwAAtAJBm9k5duyYZs+erU2bNqlDhw4X7edyuQJeG2PqtNV2qT4LFy5UeXm5cxw7dqxhgwcAAK1G0MKO1+tVaWmpUlNTFRYWprCwMOXm5uq5555TWFiYM6NTe4amtLTUOefxeFRVVaWysrKL9qmP2+1WVFRUwAEAAOwUtLAzatQo+Xw+5efnO8fQoUN1zz33KD8/X/369ZPH41FOTo7znqqqKuXm5iotLU2SlJqaqvDw8IA+xcXFKigocPoAAIC2LWhrdiIjI5WcnBzQ1rlzZ3Xv3t1pT09P19KlS5WUlKSkpCQtXbpUnTp10pQpUyRJ0dHRmjZtmubOnavu3burW7dumjdvnlJSUuoseAYAAG1TUBcoX8r8+fNVWVmp6dOnq6ysTMOGDdOOHTsUGRnp9FmxYoXCwsI0efJkVVZWatSoUdqwYYPat28fxJEDAIBQEVJhZ9euXQGvXS6XMjIylJGRcdH3dOjQQVlZWcrKymrewQEAgFYp6M/ZAQAAaE6EHQAAYDXCDgAAsBphBwAAWI2wAwAArEbYAQAAViPsAAAAqxF2AACA1Qg7AADAaoQdAABgNcIOAACwGmEHAABYjbADAACsFlK7ngPBVnP+nHw+X0Bbamqq3G53kEYEALhShB3gaypOHFXWkUp5DrkkSaeOH9Rz06W0tLQgjwwA0FiEHaCWSE+ievRPCfYwAABNhDU7AADAaoQdAABgNcIOAACwGmEHAABYjbADAACsRtgBAABWI+wAAACrEXYAAIDVCDsAAMBqPEEZ+Ab17ZUlsV8WALQmhB3gG9TeK0tivywAaG0IO8AlsFcWALRurNkBAABWI+wAAACrEXYAAIDVCDsAAMBqLFAGrpDf75fX663Tzu3pABAaCDvAFfJ6vZq1equ69urvtHF7OgCEDsIO0AS69urP7ekAEKJYswMAAKxG2AEAAFYj7AAAAKsRdgAAgNUIOwAAwGqEHQAAYDXCDgAAsBphBwAAWI2HCqLZ1N5GwefzqaYmiAMCALRJhB00m9rbKBTl/0kxA1KDPKqWUXP+nHw+X0Abe2UBQHAQdtCsvr6NwqnjB4M8mpZTceKoso5UynPIJYm9sgAgmAg7QDOJ9CSyXxYAhAAWKAMAAKsRdgAAgNUIOwAAwGqEHQAAYDXCDgAAsBphBwAAWI2wAwAArEbYAQAAViPsAAAAqxF2AACA1Qg7AADAaoQdAABgNcIOAACwWlDDzpo1azRo0CBFRUUpKipKw4cP15tvvumcN8YoIyND8fHx6tixo0aOHKnCwsKAz/D7/Zo5c6Z69Oihzp076/bbb1dRUVFLlwIAAEJUUMPO1Vdfraefflp79+7V3r179YMf/EB33HGHE2iWLVum5cuXa9WqVdqzZ488Ho/GjBmj06dPO5+Rnp6uLVu2KDs7W7t379aZM2c0YcIEVVdXB6ssAAAQQoIadiZOnKhbb71V11xzja655ho99dRT6tKli9577z0ZY7Ry5UotWrRIkyZNUnJysjZu3KgvvvhCmzdvliSVl5dr3bp1evbZZzV69GgNGTJEmzZtks/n086dO4NZGgAACBEhs2anurpa2dnZOnv2rIYPH67Dhw+rpKREY8eOdfq43W6NGDFCeXl5kiSv16tz584F9ImPj1dycrLTpz5+v18VFRUBBwAAsFNYsAfg8/k0fPhwffnll+rSpYu2bNmigQMHOmElNjY2oH9sbKw+/fRTSVJJSYkiIiIUExNTp09JSclFv2ZmZqaeeOKJJq4EuLia8+fk8/nqtKempsrtdgdhRADQdgQ97Fx77bXKz8/XqVOn9Oqrr2rq1KnKzc11zrtcroD+xpg6bbVdqs/ChQs1Z84c53VFRYUSEhIaWQFwaRUnjirrSKU8h/7x9/LU8YN6brqUlpYWxJEBgP2CHnYiIiI0YMAASdLQoUO1Z88e/epXv9KCBQskfTV7ExcX5/QvLS11Zns8Ho+qqqpUVlYWMLtTWlr6jb9A3G43/5pGi4v0JKpH/5SLnvf7/fJ6vXXamf0BgCsTMmt2LjDGyO/3KzExUR6PRzk5Oc65qqoq5ebmOkEmNTVV4eHhAX2Ki4tVUFDAv5bR6ni9Xs1avVWLtxY4x6zVW+sNQACAyxfUmZ3HHntM48ePV0JCgk6fPq3s7Gzt2rVL27Ztk8vlUnp6upYuXaqkpCQlJSVp6dKl6tSpk6ZMmSJJio6O1rRp0zR37lx1795d3bp107x585SSkqLRo0cHszTr1TcLwQzElevaq/83zv4AABouqGHnxIkTuu+++1RcXKzo6GgNGjRI27Zt05gxYyRJ8+fPV2VlpaZPn66ysjINGzZMO3bsUGRkpPMZK1asUFhYmCZPnqzKykqNGjVKGzZsUPv27YNVVptwYRaia6/+klh/AgAIXUENO+vWrfvG8y6XSxkZGcrIyLhonw4dOigrK0tZWVlNPDpcCrMQAIDWIOTW7AAAADQlwg4AALAaYQcAAFitUWGnX79+OnnyZJ32U6dOqV+/flc8KAAAgKbSqLBz5MiRencV9/v9On78+BUPCgAAoKk06G6sN954w/nv7du3Kzo62nldXV2tt956S3379m2ywQEAAFypBoWdO++8U9JXt4RPnTo14Fx4eLj69u2rZ599tskGBwAAcKUaFHZqamokSYmJidqzZ4969OjRLIMCAABoKo16qODhw4ebehwAAADNotFPUH7rrbf01ltvqbS01JnxueA///M/r3hgAAAATaFRYeeJJ57Qk08+qaFDhyouLk4ul6upxwUAANAkGhV2nn/+eW3YsEH33XdfU48HAACgSTXqOTtVVVXsbg0AAFqFRoWdBx98UJs3b27qsQAAADS5Rl3G+vLLL7V27Vrt3LlTgwYNUnh4eMD55cuXN8ngAAAArlSjws7HH3+swYMHS5IKCgoCzrFYGQAAhJJGhZ133nmnqccBAADQLBq1ZgcAAKC1aNTMzi233PKNl6vefvvtRg8IAACgKTUq7FxYr3PBuXPnlJ+fr4KCgjobhAIAAARTo8LOihUr6m3PyMjQmTNnrmhAAAAATalJ1+zce++97IsFAABCSqM3Aq3P//zP/6hDhw5N+ZEAmoDf75fX6w1oS01NldvtDtKIAKDlNCrsTJo0KeC1MUbFxcXau3ev/uM//qNJBgag6Xi9Xs1avVVde/WXJJ06flDPTRfbvgBoExoVdqKjowNet2vXTtdee62efPJJjR07tkkGBqBpde3VXz36p0iSas6fk8/nq9OH2R4ANmpU2Fm/fn1TjwNAE6nvkpXP51NNzT9eV5w4qqwjlfIc+scjJJjtAWCrK1qz4/V6tW/fPrlcLg0cOFBDhgxpqnEBaKTal6wkqSj/T4oZkBrQL9KT6Mz0AIDNGhV2SktLddddd2nXrl3q2rWrjDEqLy/XLbfcouzsbF111VVNPU4ADfD1S1bSV7M2ANBWNerW85kzZ6qiokKFhYX6/PPPVVZWpoKCAlVUVGjWrFlNPUagzbqwtiYvL885/H5/sIcFAK1Ko2Z2tm3bpp07d+pb3/qW0zZw4ED9+te/ZoEy0IRqr61hXQ0ANFyjwk5NTY3Cw8PrtIeHh6vm66sgYYXLWfCK5sPaGgC4Mo0KOz/4wQ80e/Zs/dd//Zfi4+MlScePH9ejjz6qUaNGNekAEXyXu+AVAIBQ1Kiws2rVKt1xxx3q27evEhIS5HK5dPToUaWkpGjTpk1NPUaEABa8AgBaq0aFnYSEBH344YfKycnR//3f/8kYo4EDB2r06NFNPT4AAIAr0qCw8/bbb+uRRx7Re++9p6ioKI0ZM0ZjxoyRJJWXl+v666/X888/r5tvvrlZBgsguNhjC0Br1KCws3LlSv34xz9WVFRUnXPR0dF66KGHtHz5csIOYCn22ALQGjUo7Hz00Ud65plnLnp+7Nix+uUvf3nFgwJQv1DY06r2+i0ACHUNCjsnTpyo95Zz58PCwvS3v/3tigcFoH7saQUADdegsNOrVy/5fD4NGDCg3vMff/yx4uLimmRgCJ7a6zJ4pk5o4bk7ANAwDQo7t956qxYvXqzx48erQ4cOAecqKyu1ZMkSTZgwoUkHiJZXe10Gz9QBALRmDQo7jz/+uF577TVdc801euSRR3TttdfK5XJp3759+vWvf63q6motWrSoucaKFvT1dRk8UwcA0Jo1KOzExsYqLy9PP/nJT7Rw4UIZYyRJLpdL48aN0+rVqxUbG9ssAwUAAGiMBj9UsE+fPvrjH/+osrIyffLJJzLGKCkpSTExMc0xPgAAgCvSqCcoS1JMTIxuvPHGphwLAABAk2sX7AEAAAA0J8IOAACwGmEHAABYjbADAACsRtgBAABWI+wAAACrEXYAAIDVCDsAAMBqhB0AAGC1Rj9BGYBdas6fk8/nC2hLTU2V2+0O0ogAoGkQdgBIkipOHFXWkUp5DrkkfbXb/XPTpbS0tCCPDACuDGEHgCPSk6ge/VOCPQwAaFKs2QEAAFYj7AAAAKsRdgAAgNVYswOgxfn9fnm93oA27vwC0FyCOrOTmZmpG2+8UZGRkerZs6fuvPNO7d+/P6CPMUYZGRmKj49Xx44dNXLkSBUWFgb08fv9mjlzpnr06KHOnTvr9ttvV1FRUUuWAqABvF6vZq3eqsVbC7R4a4EeyXpVGzZsUF5eXsDh9/uDPVQAFghq2MnNzdWMGTP03nvvKScnR+fPn9fYsWN19uxZp8+yZcu0fPlyrVq1Snv27JHH49GYMWN0+vRpp096erq2bNmi7Oxs7d69W2fOnNGECRNUXV0djLIAXIauvfqrR/8U9eifonbtw5S1o9AJP4u3FmjW6q11Zn8AoDGCehlr27ZtAa/Xr1+vnj17yuv16vvf/76MMVq5cqUWLVqkSZMmSZI2btyo2NhYbd68WQ899JDKy8u1bt06vfTSSxo9erQkadOmTUpISNDOnTs1bty4Fq8LQMNx2zuA5hJSC5TLy8slSd26dZMkHT58WCUlJRo7dqzTx+12a8SIEcrLy5P01XT4uXPnAvrEx8crOTnZ6VOb3+9XRUVFwAEAAOwUMmHHGKM5c+bopptuUnJysiSppKREkhQbGxvQNzY21jlXUlKiiIgIxcTEXLRPbZmZmYqOjnaOhISEpi4HAACEiJC5G+uRRx7Rxx9/rN27d9c553K5Al4bY+q01fZNfRYuXKg5c+Y4rysqKgg8QCvEXV0ALkdIhJ2ZM2fqjTfe0Lvvvqurr77aafd4PJK+mr2Ji4tz2ktLS53ZHo/Ho6qqKpWVlQXM7pSWll50Tx+3280PQ8ACF+7q6tqrvyT28wJQv6CGHWOMZs6cqS1btmjXrl1KTEwMOJ+YmCiPx6OcnBwNGTJEklRVVaXc3Fw988wzkr76V1x4eLhycnI0efJkSVJxcbEKCgq0bNmyli0IaGPq2ym9qqpKkhQREeG0Nedsy4W7ugDgYoIadmbMmKHNmzdr69atioyMdNbYREdHq2PHjnK5XEpPT9fSpUuVlJSkpKQkLV26VJ06ddKUKVOcvtOmTdPcuXPVvXt3devWTfPmzVNKSopzdxaA5lF7p3RJKsp/V2Fduskz4Ku1d8y2AAi2oIadNWvWSJJGjhwZ0L5+/Xrdf//9kqT58+ersrJS06dPV1lZmYYNG6YdO3YoMjLS6b9ixQqFhYVp8uTJqqys1KhRo7Rhwwa1b9++pUoB2qzat4yfOn5Q4dEeZlsAhIygX8a6FJfLpYyMDGVkZFy0T4cOHZSVlaWsrKwmHB0AALBByNx6DgAA0BxC4m4sAKitvsXP3FYOoDEIOwBCUu3Fzyx0BtBYhB0AIasp9stqzIMH63vP5bwPQGgi7ACwWmMePFj7PZf7PgChibADwHqNefAgDysE7MHdWAAAwGqEHQAAYDUuYwGwRn23q/t8PtXUBGlAAEICYQeANerfq+tPihmQGsRRAQg2wg4Aq9S3VxeAto01OwAAwGqEHQAAYDXCDgAAsBprdtq4+h6Lz90rQF1sTAq0XoSdNq6+x+Jz9wpQFxuTAq0XYQd1HovP3StA/ZpiY1IALY81OwAAwGqEHQAAYDXCDgAAsBphBwAAWI0FygDalPpuIZe4jRywGWEHQJtS32ah3EYO2I2wA6DN4RZyoG1hzQ4AALAaYQcAAFiNsAMAAKxG2AEAAFYj7AAAAKsRdgAAgNUIOwAAwGqEHQAAYDXCDgAAsBphBwAAWI2wAwAArEbYAQAAViPsAAAAqxF2AACA1Qg7AADAaoQdAABgNcIOAACwGmEHAABYjbADAACsRtgBAABWI+wAAACrEXYAAIDVCDsAAMBqhB0AAGA1wg4AALAaYQcAAFiNsAMAAKxG2AEAAFYj7AAAAKsRdgAAgNUIOwAAwGqEHQAAYDXCDgAAsBphBwAAWI2wAwAArEbYAQAAVgtq2Hn33Xc1ceJExcfHy+Vy6fXXXw84b4xRRkaG4uPj1bFjR40cOVKFhYUBffx+v2bOnKkePXqoc+fOuv3221VUVNSCVQAAgFAW1LBz9uxZ3XDDDVq1alW955ctW6bly5dr1apV2rNnjzwej8aMGaPTp087fdLT07VlyxZlZ2dr9+7dOnPmjCZMmKDq6uqWKgMAAISwsGB+8fHjx2v8+PH1njPGaOXKlVq0aJEmTZokSdq4caNiY2O1efNmPfTQQyovL9e6dev00ksvafTo0ZKkTZs2KSEhQTt37tS4ceNarBYAABCaQnbNzuHDh1VSUqKxY8c6bW63WyNGjFBeXp4kyev16ty5cwF94uPjlZyc7PSpj9/vV0VFRcABAADsFLJhp6SkRJIUGxsb0B4bG+ucKykpUUREhGJiYi7apz6ZmZmKjo52joSEhCYePQAACBUhG3YucLlcAa+NMXXaartUn4ULF6q8vNw5jh071iRjBQAAoSeoa3a+icfjkfTV7E1cXJzTXlpa6sz2eDweVVVVqaysLGB2p7S0VGlpaRf9bLfbLbfb3UwjDx1+v19er7dOe2pqapuoHwAAKYRndhITE+XxeJSTk+O0VVVVKTc31wkyqampCg8PD+hTXFysgoKCbww7bYXX69Ws1Vu1eGuBc8xavbXeAAQAgK2COrNz5swZffLJJ87rw4cPKz8/X926dVPv3r2Vnp6upUuXKikpSUlJSVq6dKk6deqkKVOmSJKio6M1bdo0zZ07V927d1e3bt00b948paSkOHdntXVde/VXj/4pwR4GAABBE9Sws3fvXt1yyy3O6zlz5kiSpk6dqg0bNmj+/PmqrKzU9OnTVVZWpmHDhmnHjh2KjIx03rNixQqFhYVp8uTJqqys1KhRo7Rhwwa1b9++xesBAAChJ6hhZ+TIkTLGXPS8y+VSRkaGMjIyLtqnQ4cOysrKUlZWVjOMEAAAtHYhu2YHAACgKRB2AACA1Qg7AADAaoQdAABgNcIOAACwGmEHAABYjbADAACsRtgBAABWI+wAAACrEXYAAIDVCDsAAMBqhB0AAGA1wg4AALAaYQcAAFgtLNgDQNPx+/3yer3Oa5/Pp5qaIA4IAIAQQNixiNfr1azVW9W1V39JUlH+nxQzIDXIowIAILgIO5bp2qu/evRPkSSdOn4wyKMB7FVz/px8Pl+d9tTUVLnd7iCMCMDFEHYAoBEqThxV1pFKeQ65nLZTxw/quelSWlpaEEcGoDbCDgA0UqQn0ZlJBRC6uBsLAABYjbADAACsRtgBAABWI+wAAACrEXYAAIDVCDsAAMBqhB0AAGA1wg4AALAaYQcAAFiNsAMAAKzGdhGtlN/vl9frDWjz+XyqqQnSgAAACFGEnVbK6/Vq1uqt6tqrv9NWlP8nxQxIDeKoAAAIPYSdVqxrr/4BmxCeOn4wiKMBUFt9M7CSlJqaKrfbHYQRAW0TYQcAmkl9M7Cnjh/Uc9OltLS0II4MaFsIOwDQjGrPwAJoedyNBQAArEbYAQAAVuMyFgC0oJrz5+Tz+ZzXVVVVkqSIiIiAfixiBpoOYQcAWlDFiaPKOlIpzyGXJKko/12Fdekmz4Bkpw+LmIGmRdgBgBYW6Ul0Fi2fOn5Q4dEeFjEDzYg1OwAAwGqEHQAAYDUuY4Wg+p66ymJFAPgKT6ZGQxF2QlDtp66yWBFAbfX9wq/vzq5gBoDm+ocbT6ZGQxF2QhRPXQXwTerfDDjwzq6WDAD1BRufz6e17x5UzNUDmnw8/IxEQxB2AKAVqB0mfD6fouL61dkMuDnu7Lqcy0b1h68/KWZAKqEEQUfYAYBWoHaYuBAkgvG1pfpnaWrPtpw6frBFxgdcCmEHAFqJr4eJywkStZ/WfMGl1s1cziwS0JoQdgDAUrWf1ixd3rqZYM4iAc2BsAMAFvv605oboqGzSM2pvpmmmpogDgitDmEnyC52BwP/IwNoCa3hZ9DlzDTVd8mO5+7gAsJOkH3THQwA0Nxa8mfQ5awhulj4+vqaofpmmmpfsuO5O/g6wk4I4A4GAMHUUj+DLmcN0ZWEr4ZesuNJzG0HYQcA0GIuJ5C0VPjiScxtB2GnhbHQDgBCB09ibhsIOy2MWzoBAGhZhJ0gCKVbOgEAsB1hBwDQJlzOMoLLuYW9uXZzR/Mh7DSj1vD8CgBoKy5nGcHl3MJe+3NY1Bz6CDvNiGfoAEBouZxlBA29Y6yxe5Ch5RB2mhnP0AEAu9X3/KDPj+7XQyN9Skn5x8//S10Oq6qqkiRFREQEfD6h6cpZE3ZWr16tX/ziFyouLtb111+vlStX6uabbw72sAAAbUDt2aBTxw8qa0dhgy6HFeW/q7Au3eQZkOz0uZzQhEuzIuy88sorSk9P1+rVq/W9731Pv/nNbzR+/Hj95S9/Ue/evYM9PABAG9TQy2Gnjh9UeLTnG0NTfeFHunQAasyi6vreU9/sU2sIX1aEneXLl2vatGl68MEHJUkrV67U9u3btWbNGmVmZgZ5dAAANN7XQ1Pt8HOh7euzRhe7OWbtuwcVc/WAet9Tn/rXnQbOPtX3OaF4t1qrDztVVVXyer366U9/GtA+duxY5eXl1fsev98vv9/vvC4vL5ckVVRUNOnYzp49q5NH/qLz/sp/fK3iIwqrKJc7vF29r+vvc1gffPCFzp496/QpLCzUySNHGvHZgZ/V+M9p+Bhb5+c0559hqH1OqH8v2u73tC3Vyt/fy/iedo4J+POpPufXBx98EPBnmPXqO+rUzeP0OXnkL4ruc70i///7ar+nPoWFhao+5w/8WufPyXWuymmr73Nqf/0vPi/R84sf0bBhwy76tRrrwu9tY8w3dzSt3PHjx40k8+c//zmg/amnnjLXXHNNve9ZsmSJkcTBwcHBwcFhwXHs2LFvzAqtfmbnApfLFfDaGFOn7YKFCxdqzpw5zuuamhp9/vnn6t69+0Xf0xIqKiqUkJCgY8eOKSoqKmjjaAltpda2UqfUdmptK3VK1Goj2+o0xuj06dOKj4//xn6tPuz06NFD7du3V0lJSUB7aWmpYmNj632P2+2uc+2wa9euzTXEBouKirLiL+HlaCu1tpU6pbZTa1upU6JWG9lUZ3R09CX7tLtkjxAXERGh1NRU5eTkBLTn5OTwNEsAAND6Z3Ykac6cObrvvvs0dOhQDR8+XGvXrtXRo0f18MMPB3toAAAgyKwIOz/60Y908uRJPfnkkyouLlZycrL++Mc/qk+fPsEeWoO43W4tWbIk5J9X0BTaSq1tpU6p7dTaVuqUqNVGbaXO2lzGXOp+LQAAgNar1a/ZAQAA+CaEHQAAYDXCDgAAsBphBwAAWI2w08IyMzN14403KjIyUj179tSdd96p/fv3B/QxxigjI0Px8fHq2LGjRo4cqcLCwiCNuPHWrFmjQYMGOQ+vGj58uN58803nvC111paZmSmXy6X09HSnzZZaMzIy5HK5Ag6P5x/779hS5wXHjx/Xvffeq+7du6tTp04aPHhwwAaHNtTbt2/fOt9Tl8ulGTNmSLKjxgvOnz+vxx9/XImJierYsaP69eunJ598UjU1NU4fW+o9ffq00tPT1adPH3Xs2FFpaWnas2ePc96WOi/bFW1MhQYbN26cWb9+vSkoKDD5+fnmtttuM7179zZnzpxx+jz99NMmMjLSvPrqq8bn85kf/ehHJi4uzlRUVARx5A33xhtvmD/84Q9m//79Zv/+/eaxxx4z4eHhpqCgwBhjT51f98EHH5i+ffuaQYMGmdmzZzvtttS6ZMkSc/3115vi4mLnKC0tdc7bUqcxxnz++eemT58+5v777zfvv/++OXz4sNm5c6f55JNPnD421FtaWhrw/czJyTGSzDvvvGOMsaPGC37+85+b7t27m9///vfm8OHD5r//+79Nly5dzMqVK50+ttQ7efJkM3DgQJObm2sOHDhglixZYqKiokxRUZExxp46LxdhJ8hKS0uNJJObm2uMMaampsZ4PB7z9NNPO32+/PJLEx0dbZ5//vlgDbPJxMTEmN/+9rdW1nn69GmTlJRkcnJyzIgRI5ywY1OtS5YsMTfccEO952yq0xhjFixYYG666aaLnret3gtmz55t+vfvb2pqaqyr8bbbbjMPPPBAQNukSZPMvffea4yx53v6xRdfmPbt25vf//73Ae033HCDWbRokTV1NgSXsYKsvLxcktStWzdJ0uHDh1VSUqKxY8c6fdxut0aMGKG8vLygjLEpVFdXKzs7W2fPntXw4cOtrHPGjBm67bbbNHr06IB222o9cOCA4uPjlZiYqLvuukuHDh2SZF+db7zxhoYOHaof/vCH6tmzp4YMGaIXXnjBOW9bvZJUVVWlTZs26YEHHpDL5bKuxptuuklvvfWW/vrXv0qSPvroI+3evVu33nqrJHu+p+fPn1d1dbU6dOgQ0N6xY0ft3r3bmjobgrATRMYYzZkzRzfddJOSk5MlydnQtPYmprGxsXU2O20NfD6funTpIrfbrYcfflhbtmzRwIEDraszOztbH374oTIzM+ucs6nWYcOG6cUXX9T27dv1wgsvqKSkRGlpaTp58qRVdUrSoUOHtGbNGiUlJWn79u16+OGHNWvWLL344ouS7Pq+XvD666/r1KlTuv/++yXZV+OCBQt0991367rrrlN4eLiGDBmi9PR03X333ZLsqTcyMlLDhw/Xz372M3322Weqrq7Wpk2b9P7776u4uNiaOhvCiu0iWqtHHnlEH3/8sXbv3l3nnMvlCnhtjKnT1hpce+21ys/P16lTp/Tqq69q6tSpys3Ndc7bUOexY8c0e/Zs7dixo86/pL7OhlrHjx/v/HdKSoqGDx+u/v37a+PGjfrud78ryY46JammpkZDhw7V0qVLJUlDhgxRYWGh1qxZo3/5l39x+tlSryStW7dO48ePV3x8fEC7LTW+8sor2rRpkzZv3qzrr79e+fn5Sk9PV3x8vKZOner0s6Hel156SQ888IB69eql9u3b69vf/ramTJmiDz/80OljQ52Xi5mdIJk5c6beeOMNvfPOO7r66qud9gt3ttRO16WlpXVSeGsQERGhAQMGaOjQocrMzNQNN9ygX/3qV1bV6fV6VVpaqtTUVIWFhSksLEy5ubl67rnnFBYW5tRjQ621de7cWSkpKTpw4IBV31NJiouL08CBAwPavvWtb+no0aOS7Pt/9dNPP9XOnTv14IMPOm221fjv//7v+ulPf6q77rpLKSkpuu+++/Too486M7I21du/f3/l5ubqzJkzOnbsmD744AOdO3dOiYmJVtV5uQg7LcwYo0ceeUSvvfaa3n77bSUmJgacv/AXMScnx2mrqqpSbm6u0tLSWnq4Tc4YI7/fb1Wdo0aNks/nU35+vnMMHTpU99xzj/Lz89WvXz9raq3N7/dr3759iouLs+p7Kknf+9736jwW4q9//auzwbBt9a5fv149e/bUbbfd5rTZVuMXX3yhdu0Cf+21b9/eufXctnqlr/5BEhcXp7KyMm3fvl133HGHlXVeUpAWRrdZP/nJT0x0dLTZtWtXwO2eX3zxhdPn6aefNtHR0ea1114zPp/P3H333a3ylsCFCxead9991xw+fNh8/PHH5rHHHjPt2rUzO3bsMMbYU2d9vn43ljH21Dp37lyza9cuc+jQIfPee++ZCRMmmMjISHPkyBFjjD11GvPVYwTCwsLMU089ZQ4cOGBefvll06lTJ7Np0yanjy31VldXm969e5sFCxbUOWdLjcYYM3XqVNOrVy/n1vPXXnvN9OjRw8yfP9/pY0u927ZtM2+++aY5dOiQ2bFjh7nhhhvMd77zHVNVVWWMsafOy0XYaWGS6j3Wr1/v9KmpqTFLliwxHo/HuN1u8/3vf9/4fL7gDbqRHnjgAdOnTx8TERFhrrrqKjNq1Cgn6BhjT531qR12bKn1wrM4wsPDTXx8vJk0aZIpLCx0zttS5wW/+93vTHJysnG73ea6664za9euDThvS73bt283ksz+/fvrnLOlRmOMqaioMLNnzza9e/c2HTp0MP369TOLFi0yfr/f6WNLva+88orp16+fiYiIMB6Px8yYMcOcOnXKOW9LnZfLZYwxQZxYAgAAaFas2QEAAFYj7AAAAKsRdgAAgNUIOwAAwGqEHQAAYDXCDgAAsBphBwAAWI2wAwAArEbYAQAAViPsAAAAqxF2AACA1Qg7AFqlbdu26aabblLXrl3VvXt3TZgwQQcPHnTO5+XlafDgwerQoYOGDh2q119/XS6XS/n5+U6fv/zlL7r11lvVpUsXxcbG6r777tPf//73IFQDoDkRdgC0SmfPntWcOXO0Z88evfXWW2rXrp3+6Z/+STU1NTp9+rQmTpyolJQUffjhh/rZz36mBQsWBLy/uLhYI0aM0ODBg7V3715t27ZNJ06c0OTJk4NUEYDmwq7nAKzwt7/9TT179pTP59Pu3bv1+OOPq6ioSB06dJAk/fa3v9WPf/xj/e///q8GDx6sxYsX6/3339f27dudzygqKlJCQoL279+va665JlilAGhizOwAaJUOHjyoKVOmqF+/foqKilJiYqIk6ejRo9q/f78GDRrkBB1J+s53vhPwfq/Xq3feeUddunRxjuuuu875bAD2CAv2AACgMSZOnKiEhAS98MILio+PV01NjZKTk1VVVSVjjFwuV0D/2pPYNTU1mjhxop555pk6nx0XF9esYwfQsgg7AFqdkydPat++ffrNb36jm2++WZK0e/du5/x1112nl19+WX6/X263W5K0d+/egM/49re/rVdffVV9+/ZVWBg/CgGbcRkLQKsTExOj7t27a+3atfrkk0/09ttva86cOc75KVOmqKamRv/2b/+mffv2afv27frlL38pSc6Mz4wZM/T555/r7rvv1gcffKBDhw5px44deuCBB1RdXR2UugA0D8IOgFanXbt2ys7OltfrVXJysh599FH94he/cM5HRUXpd7/7nfLz8zV48GAtWrRIixcvliRnHU98fLz+/Oc/q7q6WuPGjVNycrJmz56t6OhotWvHj0bAJtyNBaBNePnll/Wv//qvKi8vV8eOHYM9HAAtiAvVAKz04osvql+/furVq5c++ugjLViwQJMnTyboAG0QYQeAlUpKSrR48WKVlJQoLi5OP/zhD/XUU08Fe1gAgoDLWAAAwGqswgMAAFYj7AAAAKsRdgAAgNUIOwAAwGqEHQAAYDXCDgAAsBphBwAAWI2wAwAArPb/AMg8pJwxgTMfAAAAAElFTkSuQmCC",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Distribution of Age\n",
    "sns.histplot(bank.age, bins=100)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 449
    },
    "id": "xbf-gTVobB0-",
    "outputId": "2b6ad875-dd43-4b22-ab4c-5367d5a735d2",
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAhEAAAGwCAYAAAAXNjfEAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjcuMiwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8pXeV/AAAACXBIWXMAAA9hAAAPYQGoP6dpAAAfLUlEQVR4nO3df5CV5X3w/89ZdtmFZVl+iCwIqKkxURf4PiwEIRlDNUVJNCbOtP5YLE5sUk21EMDQxCFQbapNqikdm6SNNtGvNkwyiJM2JaOdCklmkViUBn/EMI0GUyE0FGQVAWWv54/Mnodld9ndi4XdPbxeMzuz5z73fZ/rui/gvDnnLBRSSikAAHqorK8HAAAMTCICAMgiIgCALCICAMgiIgCALCICAMgiIgCALOW5B7a0tMRrr70WNTU1USgUenNMAMAJklKK5ubmGD9+fJSVHd9rCdkR8dprr8XEiROP68EBgL7x6quvxoQJE47rHNkRUVNTUxzE8OHDj2sQAMDJsW/fvpg4cWLxefx4ZEdE61sYw4cPFxEAMMD0xkcRfLASAMgiIgCALCICAMgiIgCALCICAMgiIgCALCICAMgiIgCALCICAMgiIgCALCICAMgiIgCALCICAMgiIgCALCICAMgiIgCALCICAMgiIgCALCICAMgiIgCALCICAMgiIgCALCICAMgiIgCALCICAMgiIgCALCICAMgiIgCALCICAMgiIgCALCICAMgiIgCALOV9PYDelFKKAwcOZB138ODBiIiorKyMQqHQ20PrVFVV1Ul9PADoLSUVEQcOHIh58+b19TB6ZN26dTFkyJC+HgYA9Ji3MwCALCX1SsSR3vj/ro1U1s3pHX47av5zdURENE+9JmJQxQkcWUSh5Z0YtuXbJ/QxAOBEK9mISGXleTEwqOKER0Q6oWcHgJPD2xkAQBYRAQBkEREAQBYRAQBkEREAQBYRAQBkEREAQBYRAQBkEREAQBYRAQBkEREAQBYRAQBkEREAQBYRAQBkEREAQBYRAQBkEREAQBYRAQBkEREAQBYRAQBkEREAQBYRAQBkEREAQBYRAQBkEREAQBYRAQBkEREAQBYRAQBkEREAQBYRAQBkEREAQBYRAQBkEREAQBYRAQBkEREAQBYRAQBkEREAQBYRAQBkEREAQBYRAQBkEREAQBYRAQBkEREAQBYRAQBkEREAQBYRAQBkEREAQBYRAQBkEREAQBYRAQBkEREAQBYRAQBkEREAQBYRAQBkEREAQBYRAQBkEREAQBYRAQBkEREAQBYRAQBkEREAQBYRAQBkEREAQBYRAQBkEREAQBYRAQBkEREAQBYRAQBkEREAQJbyvh7AkVJKceDAgYiIqKqqikKh0McjojdZX4DS0q9eiThw4EDMmzcv5s2bV3yyoXRYX4DS0q8iAgAYOEQEAJBFRAAAWUQEAJBFRAAAWUQEAJBFRAAAWUQEAJBFRAAAWUQEAJBFRAAAWUQEAJBFRAAAWUQEAJBFRAAAWUQEAJBFRAAAWUQEAJBFRAAAWUQEAJBFRAAAWUQEAJBFRAAAWUQEAJBFRAAAWUQEAJBFRAAAWUQEAJBFRAAAWUQEAJBFRAAAWUQEAJBFRAAAWUQEAJBFRAAAWUQEAJBFRAAAWUQEAJBFRAAAWUQEAJBFRAAAWUQEAJBFRAAAWUQEAJBFRAAAWUQEAJBFRAAAWUQEAJBFRAAAWUQEAJBFRAAAWUQEAJBFRAAAWUQEAJBFRAAAWUQEAJBFRAAAWUQEAJBFRAAAWUQEAJBFRAAAWUQEAJBFRAAAWUQEAJBFRAAAWUQEAJBFRAAAWUQEAJBFRAAAWcr7egCcmubNm9fXQ+h3KioqYvDgwfHWW29FS0tLRERUVlbG22+/HS0tLVFWVlbc3mrOnDmxcuXKaGpqij//8z+PgwcPRkTEpEmTYvv27VEoFOKCCy6I559/PoYOHRozZsyIDRs2REopIiLKy8vjjjvuiNmzZ8cDDzwQjzzySDQ2NsZ5550Xq1ativPOO6/N/vX19XHfffdFU1NTfPGLX4w333yzOI65c+fGl770pYiIuOKKK+LRRx+N/fv3x/z58+PGG2885tybmppi1apVsXDhwpg9e3aX2zu6PyLa7dvR8Udva2pqKo77s5/9bLvH6WoMXc2lq9sdeeCBB+Lhhx+OoUOHxu23396txz2RWsc8d+7c+Od//ueI6PhaHb1/d9atu9fkVDNQrkkhtf7p0EP79u2L2traeP3112P48OG9Mpi33nqr+OSybt26GDJkSPbxzdOujxhU0b0DD78dNc/8/z0/LtcRj5czz4Hq5z//eXzqU5/q62GUnIceeigWLVoU//u//5t1/IgRI+Lv//7v49prr42WlpYoFAoxcuTITs/3jW98I5YtW9bu/pEjR8aePXva7V8oFGLt2rUxYsSIDs934MCBmD9/fvzmN7+J0047LR5++OGoqqrqdHtHx40ePToiInbv3l3cNyLaHX/0tvvvvz9uvPHG2L17d0REjB49Oh555JHi43Q1hq7mcv/998cf/dEfdXq7o/Pt3bs3Pv7xjxfD7egxnWxHzqlQKHQ5rp6sW3evyammp7/ueqo3n7+9ncFJ0/q3RXrXTTfdlB0QEb990rrllluKr3KklI55vptvvrnD+zsKiNbzfeELX+j0fI888kjxSXz37t3xT//0T8fc3tlxR+/b0fFHb1u+fHnxdkeP09UYuprLkefv6HZH51u+fHkc+Xe77jzuiXTknLozrp6uW3euyammp7/u+lK/eiVi//798eEPfzgiItauXdvj8jpw4EB8/OMfj4iI5v8zP6J8cPcOPNmvRLxzKGqe/e3finLmORB997vfjX/8x3/s62HQh/76r/86pk+f3mbbr371q1iwYEEcPny4uK28vDzuvvvuWLZsWbvt3/rWt2LChAkdHnekQYMGRUqpzds/gwYNiojo9Jgj93vwwQcjIjocW+sYjtbVmDpy9Pn+4z/+I5YuXdrpmDp63BOpO9f5yHF1tp7dXbej9z8VdXUNe0NvPn93OyIOHjxYfL+1dRATJ07s1YjYs2dPMQKOV/PUayIGD+3ezic7Ig7tj5r/XH1iHwP6meHDh8djjz0WZWW/fQE0pRSf/exn45lnnmnzB+agQYOiuro63njjjXYRMG3atPirv/qrWLZsWbvjelNDQ0OUlZV1OLZp06bFl770pSgUCsXtnc2lK0eeL6UUV155ZTQ3N3e474wZM9o97onU3Tm1jisiOl3PnqxbZ9f4VHCs3xO9eU365O2Mu+66K2pra4tfEydOPK4HBk4t+/bti02bNhVvb9++PZ5++ul2TyiHDx+Offv2tfsQ6eHDh+Ppp5+OTZs2dXhcb9q8eXOnY3v66adj+/btbbZ3NpeuHHm+TZs2dRoQEdHh455I3Z1T67iOtZ49WbfOrvGpoKtr2B+vSbd/OuNzn/tcLF68uHi79ZWI3lRZWVn8/njfzoiyfvyDJ0eM7VR4O+Odd96JK664oq+HQR8bPnx4zJw5s3h70qRJMWPGjB69EtHQ0BAzZ87s8LjeNH369CgUCh2OraGhISZNmtRm/87m0pUjzzdx4sSoqak55isRRz/uidTdOb3vfe8rjquz9ezJunV2jU8Fx/o90V+vSbdfiaisrIzhw4e3+eptR75MU1VVFUOGDOnRV5sn4/78MthxznOgfdXU1MQf//Ef9+EFpz9YuXJl8a2MiN/+fu/ow7aFQiFWrFjR7mXb1v3Lysq6/JDuoEGD2jxW67bWz0V0dexnPvOZTse2cOHCTsfWU0eer6ysLFasWNHpmBYtWnRSX97vzpwGDRpUHP+x1rO763bk/qfaWxkRx/490V+viZ/O4KS49tprT5kfZT3Zhg7t5md/jmHMmDHd3re8vOev8k2ZMiWmTZvWbvuECRPiuuuuK/7hWCgU4rrrrouGhoYOt59xxhkdHnekQqEQjY2N0djY2Ob4xsbGduecPHlyu+MbGxvjjDPO6HRsrWPozlwmT558zNtHn2/69OnHHNPJdqzrHNF+XF1ds+5co2Nd41NBT3/d9TURwUmzatWqvh5CSfr6178eo0aNyj5+5MiRcd999xX/5l5WVnbM833ta1/r8P6RI0d2uH9ZWVnccccdnZ6vsbGx+O88nHbaaXHdddcdc3tnxx29b0fHH73tzjvvLN7u6HG6GkNXczny/B3d7uh8d955Z5sn7dGjR3f5uCfSkXM6clydjb+n69ada3Kq6emvu74kIjhpfBj32CoqKqK6urrNy/CVlZVtntyPNmfOnJg0aVIsXbq0zWeKWt87LRQKUV9fH4VCIaqrq2POnDltngjKy8vjtttui7Fjx0ZjY2OUlZVFY2NjLF26NMaOHdtu//r6+nj3u98dS5cujerq6jbjuO2222LEiBExYsSIuP7666O6urr4CkBn/9BUxG/f0lu8eHGMHTs2PvOZzxTfluxse0fHLV68OJYsWdJm346OP3rbiBEjYsmSJcVxL168uM3jdDWGrubSes7Obnd0vhEjRsT8+fOLa7ZkyZI+/dzUkXOaP39+p9eqo/27WrfuXpNTTU9/3fWlfvXvRPgXK0vb8a4vAMfPv1gJAPQ5EQEAZBERAEAWEQEAZBERAEAWEQEAZBERAEAWEQEAZBERAEAWEQEAZBERAEAWEQEAZBERAEAWEQEAZBERAEAWEQEAZBERAEAWEQEAZBERAEAWEQEAZBERAEAWEQEAZBERAEAWEQEAZBERAEAWEQEAZBERAEAWEQEAZBERAEAWEQEAZBERAEAWEQEAZBERAEAWEQEAZBERAEAWEQEAZBERAEAWEQEAZBERAEAWEQEAZBERAEAWEQEAZBERAEAWEQEAZBERAEAWEQEAZBERAEAWEQEAZBERAEAWEQEAZBERAEAWEQEAZBERAEAWEQEAZBERAEAWEQEAZBERAEAWEQEAZBERAEAWEQEAZBERAEAWEQEAZBERAEAWEQEAZBERAEAWEQEAZBERAEAWEQEAZCnv6wEcqaqqKtatW1f8ntJifQFKS7+KiEKhEEOGDOnrYXCCWF+A0uLtDAAgi4gAALKICAAgi4gAALKICAAgi4gAALKICAAgi4gAALKICAAgi4gAALKICAAgi4gAALKICAAgi4gAALKICAAgi4gAALKICAAgi4gAALKICAAgi4gAALKICAAgi4gAALKICAAgi4gAALKICAAgi4gAALKICAAgi4gAALKICAAgi4gAALKICAAgi4gAALKICAAgi4gAALKICAAgi4gAALKICAAgi4gAALKICAAgi4gAALKICAAgi4gAALKICAAgi4gAALKICAAgi4gAALKICAAgi4gAALKICAAgi4gAALKICAAgi4gAALKICAAgi4gAALKICAAgi4gAALKICAAgi4gAALKICAAgi4gAALKICAAgi4gAALKICAAgi4gAALKICAAgi4gAALKICAAgi4gAALKICAAgS3lfD+BEKbS8E6m7Ox9+u+PvT5BCyzsn/DEA4EQr2YgYtuXbWcfV/OfqXh4JAJQmb2cAAFlK6pWIqqqqWLduXY+PSynFwYMHIyKisrIyCoVCbw+tU1VVVSftsQCgN5VURBQKhRgyZEjWsUOHDu3l0QBAafN2BgCQRUQAAFlEBACQRUQAAFlEBACQRUQAAFlEBACQRUQAAFlEBACQRUQAAFlEBACQRUQAAFlEBACQRUQAAFlEBACQRUQAAFlEBACQRUQAAFlEBACQRUQAAFlEBACQRUQAAFlEBACQRUQAAFlEBACQRUQAAFlEBACQRUQAAFlEBACQRUQAAFlEBACQRUQAAFlEBACQpTz3wJRSRETs27ev1wYDAJxYrc/brc/jxyM7IpqbmyMiYuLEicc9CADg5Gpubo7a2trjOkchZaZIS0tLvPbaa1FTUxOFQuG4BnGkffv2xcSJE+PVV1+N4cOH99p5+xvzLD2nylzNs7SYZ2npzjxTStHc3Bzjx4+PsrLj+1RD9isRZWVlMWHChON68GMZPnx4SS90K/MsPafKXM2ztJhnaelqnsf7CkQrH6wEALKICAAgS7+LiMrKylixYkVUVlb29VBOKPMsPafKXM2ztJhnaTnZ88z+YCUAcGrrd69EAAADg4gAALKICAAgi4gAALL0u4j46le/GmeffXZUVVVFQ0ND/OhHP+rrIXXbypUro1AotPmqq6sr3p9SipUrV8b48eNjyJAhMWfOnHj++efbnOPgwYNx6623xmmnnRbV1dXx0Y9+NH71q1+d7Km08cMf/jCuuOKKGD9+fBQKhXjsscfa3N9b89qzZ09cf/31UVtbG7W1tXH99dfH3r17T/Ds/p+u5nnDDTe0W98LL7ywzT4DYZ533XVXzJgxI2pqauL000+Pj33sY/HSSy+12acU1rQ78yyFNf3a174WU6ZMKf7jQrNmzYp169YV7y+FtYzoep6lsJYdueuuu6JQKMSiRYuK2/rVmqZ+ZPXq1amioiJ94xvfSC+88EJauHBhqq6uTr/85S/7emjdsmLFinTBBRekHTt2FL927dpVvP/uu+9ONTU1ac2aNWnr1q3p6quvTuPGjUv79u0r7nPTTTelM844Iz3xxBPpmWeeSb/7u7+bpk6dmt55552+mFJKKaV//dd/Tbfffntas2ZNioi0du3aNvf31rwuu+yyVF9fn5qamlJTU1Oqr69Pl19++cmaZpfzXLBgQbrsssvarO/u3bvb7DMQ5nnppZemb37zm+m5555LW7ZsSR/5yEfSpEmT0htvvFHcpxTWtDvzLIU1/d73vpe+//3vp5deeim99NJL6fOf/3yqqKhIzz33XEqpNNayO/MshbU82k9+8pN01llnpSlTpqSFCxcWt/enNe1XEfG+970v3XTTTW22vfe9701/9md/1kcj6pkVK1akqVOndnhfS0tLqqurS3fffXdx24EDB1JtbW36+te/nlJKae/evamioiKtXr26uM9///d/p7KysvSDH/zghI69u45+cu2teb3wwgspItJTTz1V3Gfjxo0pItLPfvazEzyr9jqLiCuvvLLTYwbiPFNKadeuXSki0oYNG1JKpbumR88zpdJd05EjR6b777+/ZNeyVes8Uyq9tWxubk7vfve70xNPPJE++MEPFiOiv61pv3k749ChQ7F58+aYO3dum+1z586NpqamPhpVz23bti3Gjx8fZ599dlxzzTXxi1/8IiIiXn755di5c2eb+VVWVsYHP/jB4vw2b94cb7/9dpt9xo8fH/X19f32GvTWvDZu3Bi1tbUxc+bM4j4XXnhh1NbW9qu5r1+/Pk4//fQ499xz45Of/GTs2rWreN9Anefrr78eERGjRo2KiNJd06Pn2aqU1vTw4cOxevXqePPNN2PWrFklu5ZHz7NVKa3ln/zJn8RHPvKR+NCHPtRme39b0+z/gKu3/eY3v4nDhw/H2LFj22wfO3Zs7Ny5s49G1TMzZ86Mhx56KM4999z49a9/HX/xF38Rs2fPjueff744h47m98tf/jIiInbu3BmDBw+OkSNHttunv16D3prXzp074/TTT293/tNPP73fzH3evHnx+7//+3HmmWfGyy+/HMuXL4+LL744Nm/eHJWVlQNynimlWLx4cXzgAx+I+vr6iCjNNe1onhGls6Zbt26NWbNmxYEDB2LYsGGxdu3aOP/884tPBqWylp3NM6J01jIiYvXq1fHMM8/E008/3e6+/vb7s99ERKuj/1vxlFKv/lfjJ9K8efOK30+ePDlmzZoVv/M7vxMPPvhg8QM+OfMbCNegN+bV0f79ae5XX3118fv6+vqYPn16nHnmmfH9738/rrrqqk6P68/zvOWWW+KnP/1p/PjHP253XymtaWfzLJU1fc973hNbtmyJvXv3xpo1a2LBggWxYcOGTsc3UNeys3mef/75JbOWr776aixcuDAef/zxqKqq6nS//rKm/ebtjNNOOy0GDRrUroB27drVrrgGiurq6pg8eXJs27at+FMax5pfXV1dHDp0KPbs2dPpPv1Nb82rrq4ufv3rX7c7///8z//027mPGzcuzjzzzNi2bVtEDLx53nrrrfG9730vnnzyyZgwYUJxe6mtaWfz7MhAXdPBgwfHOeecE9OnT4+77rorpk6dGqtWrSq5texsnh0ZqGu5efPm2LVrVzQ0NER5eXmUl5fHhg0b4m//9m+jvLy8OI7+sqb9JiIGDx4cDQ0N8cQTT7TZ/sQTT8Ts2bP7aFTH5+DBg/Hiiy/GuHHj4uyzz466uro28zt06FBs2LChOL+GhoaoqKhos8+OHTviueee67fXoLfmNWvWrHj99dfjJz/5SXGfTZs2xeuvv95v57579+549dVXY9y4cRExcOaZUopbbrklHn300fj3f//3OPvss9vcXypr2tU8OzJQ1/RoKaU4ePBgyaxlZ1rn2ZGBupaXXHJJbN26NbZs2VL8mj59ejQ2NsaWLVviXe96V/9a025/BPMkaP0RzwceeCC98MILadGiRam6ujq98sorfT20blmyZElav359+sUvfpGeeuqpdPnll6eampri+O++++5UW1ubHn300bR169Z07bXXdvhjORMmTEj/9m//lp555pl08cUX9/mPeDY3N6dnn302Pfvssyki0r333pueffbZ4o/e9ta8LrvssjRlypS0cePGtHHjxjR58uST+qNVx5pnc3NzWrJkSWpqakovv/xyevLJJ9OsWbPSGWecMeDmefPNN6fa2tq0fv36Nj8Ot3///uI+pbCmXc2zVNb0c5/7XPrhD3+YXn755fTTn/40ff7zn09lZWXp8ccfTymVxlp2Nc9SWcvOHPnTGSn1rzXtVxGRUkp/93d/l84888w0ePDgNG3atDY/jtXftf6sbkVFRRo/fny66qqr0vPPP1+8v6WlJa1YsSLV1dWlysrKdNFFF6WtW7e2Ocdbb72VbrnlljRq1Kg0ZMiQdPnll6ft27ef7Km08eSTT6aIaPe1YMGClFLvzWv37t2psbEx1dTUpJqamtTY2Jj27NlzkmZ57Hnu378/zZ07N40ZMyZVVFSkSZMmpQULFrSbw0CYZ0dzjIj0zW9+s7hPKaxpV/MslTX9xCc+Ufwzc8yYMemSSy4pBkRKpbGWKR17nqWylp05OiL605r6r8ABgCz95jMRAMDAIiIAgCwiAgDIIiIAgCwiAgDIIiIAgCwiAgDIIiIAgCwiAgawOXPmxKJFi/rksdevXx+FQiH27t3bJ48P9D0RAXSpo1iZPXt27NixI2pra/tmUECfExFwCnv77bezjx08eHDU1dVFoVDoxREBA4mIgAHizTffjD/8wz+MYcOGxbhx4+Kee+5pc3+hUIjHHnuszbYRI0bEt771rYiIeOWVV6JQKMR3vvOdmDNnTlRVVcXDDz8cu3fvjmuvvTYmTJgQQ4cOjcmTJ8e3v/3t4jluuOGG2LBhQ6xatSoKhUIUCoV45ZVXOnw7Y82aNXHBBRdEZWVlnHXWWe3GeNZZZ8Vf/uVfxic+8YmoqamJSZMmxT/8wz/06nUCTh4RAQPEbbfdFk8++WSsXbs2Hn/88Vi/fn1s3ry5x+dZtmxZ/Omf/mm8+OKLcemll8aBAweioaEh/uVf/iWee+65+NSnPhXXX399bNq0KSIiVq1aFbNmzYpPfvKTsWPHjtixY0dMnDix3Xk3b94cf/AHfxDXXHNNbN26NVauXBnLly8vRkyre+65J6ZPnx7PPvtsfPrTn46bb745fvazn2VdE6Bvlff1AICuvfHGG/HAAw/EQw89FL/3e78XEREPPvhgTJgwocfnWrRoUVx11VVtti1durT4/a233ho/+MEP4rvf/W7MnDkzamtrY/DgwTF06NCoq6vr9Lz33ntvXHLJJbF8+fKIiDj33HPjhRdeiC9/+ctxww03FPf78Ic/HJ/+9Kcj4rdB85WvfCXWr18f733ve3s8F6BveSUCBoD/+q//ikOHDsWsWbOK20aNGhXvec97enyu6dOnt7l9+PDh+OIXvxhTpkyJ0aNHx7Bhw+Lxxx+P7du39+i8L774Yrz//e9vs+39739/bNu2LQ4fPlzcNmXKlOL3hUIh6urqYteuXT2eB9D3vBIBA0BKqct9CoVCu/06+uBkdXV1m9v33HNPfOUrX4m/+Zu/icmTJ0d1dXUsWrQoDh061OMxHv0hy47GXVFR0W7cLS0tPXosoH/wSgQMAOecc05UVFTEU089Vdy2Z8+e+PnPf168PWbMmNixY0fx9rZt22L//v1dnvtHP/pRXHnllTF//vyYOnVqvOtd74pt27a12Wfw4MFtXk3oyPnnnx8//vGP22xramqKc889NwYNGtTlOICBxysRMAAMGzYsbrzxxrjtttti9OjRMXbs2Lj99tujrOz//T3g4osvjvvuuy8uvPDCaGlpiWXLlrX7W39HzjnnnFizZk00NTXFyJEj4957742dO3fGeeedV9znrLPOik2bNsUrr7wSw4YNi1GjRrU7z5IlS2LGjBlx5513xtVXXx0bN26M++67L7761a/2zkUA+h2vRMAA8eUvfzkuuuii+OhHPxof+tCH4gMf+EA0NDQU77/nnnti4sSJcdFFF8V1110XS5cujaFDh3Z53uXLl8e0adPi0ksvjTlz5kRdXV187GMfa7PP0qVLY9CgQXH++efHmDFjOvy8xLRp0+I73/lOrF69Ourr6+MLX/hC3HHHHW0+VAmUlkLqzputAABH8UoEAJBFRAAAWUQEAJBFRAAAWUQEAJBFRAAAWUQEAJBFRAAAWUQEAJBFRAAAWUQEAJDl/wKhQFRbeMRJvAAAAABJRU5ErkJggg==",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Boxplot for 'duration'\n",
    "g = sns.boxplot(x=bank[\"duration\"])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 466
    },
    "id": "M8K1xvG6bGAI",
    "outputId": "13f4750a-53cd-4599-80ef-092489cd9fde",
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<Axes: xlabel='duration', ylabel='Count'>"
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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4oknfDSi87NgwQJz1VVXNXistrbWuFwu88wzz1ht3333nXE6neall14yxhhz7NgxExoaavLy8qw+hw8fNq1atTIbNmy4qGO3q26YaK66du7caSSZjz76yOqzdetWI8l88cUXF7mq+hoLTXfeeWejnwnEOo0xpry83EgyBQUFxpjgPad16zQmeM9pVFSU+fd///egPZdnna3TmOA7l8ePHzdJSUkmPz/fDB061ApN/nZOuTznIzU1NXK73Ro1apRX+6hRo1RYWOijUZ2/PXv2KD4+XomJibrnnnv01VdfSZL27dunsrIyr/rCw8M1dOhQqz63263Tp0979YmPj1dycrLf/g2aq66tW7fK6XRq0KBBVp/rrrtOTqfTr2rfvHmzOnXqpD59+mjKlCkqLy+3jgVqnZWVlZKk6OhoScF7TuvWeVYwndMzZ84oLy9PJ0+e1ODBg4P2XNat86xgOpcPP/ywbr/9do0YMcKr3d/OaYvZEdzf/O///q/OnDlT74HBcXFx9R4s7K8GDRqkV155RX369NGRI0f01FNPaciQIdqxY4dVQ0P1ff3115KksrIyhYWFKSoqql4ff/0bNFddZWVl6tSpU73v79Spk9/Ufuutt+qXv/ylunfvrn379mn+/Pm6+eab5Xa7FR4eHpB1GmM0a9Ys3XDDDUpOTpYUnOe0oTql4DmnxcXFGjx4sL777ju1b99e69evV9++fa1/+QXLuWysTil4zqUk5eXl6bPPPtO2bdvqHfO3/34SmnzM4XB4vTfG1GvzV7feeqv1zykpKRo8eLB69eql1atXWwsSm1JfIPwNmqOuhvr7U+1333239c/JyckaMGCAunfvrr/85S8aO3Zso5/z5zqnT5+uzz//XFu2bKl3LJjOaWN1Bss5vfzyy1VUVKRjx45p3bp1mjRpkgoKChodX6Cey8bq7Nu3b9Ccy4MHD+qRRx7Rxo0b1aZNm0b7+cs55fKcj8TGxqp169b1Em55eXm9RB0oIiIilJKSoj179lh30f1cfS6XSzU1NaqoqGi0j79prrpcLpeOHDlS7/u//fZbv629c+fO6t69u/bs2SMp8OqcMWOG3nrrLb3//vvq2rWr1R5s57SxOhsSqOc0LCxMvXv31oABA5SVlaWrrrpKzz33XNCdy8bqbEignku3263y8nKlpqYqJCREISEhKigo0PPPP6+QkBBrHP5yTglNPhIWFqbU1FTl5+d7tefn52vIkCE+GtWF8Xg82rVrlzp37qzExES5XC6v+mpqalRQUGDVl5qaqtDQUK8+paWlKikp8du/QXPVNXjwYFVWVuqTTz6x+nz88ceqrKz029qPHj2qgwcPqnPnzpICp05jjKZPn67XX39d7733nhITE72OB8s5PVedDQnUc1qXMUYejydozmVjztbZkEA9l8OHD1dxcbGKioqs14ABAzRhwgQVFRWpZ8+e/nVObS8ZR7M7u+XAihUrzM6dO016erqJiIgw+/fv9/XQbJk9e7bZvHmz+eqrr8xHH31k0tLSTGRkpDX+Z555xjidTvP666+b4uJic++99zZ4m2jXrl3Npk2bzGeffWZuvvlmn285cPz4cbN9+3azfft2I8lkZ2eb7du3W1tBNFddt9xyi+nfv7/ZunWr2bp1q0lJSbmkt/r+XJ3Hjx83s2fPNoWFhWbfvn3m/fffN4MHDzZdunQJuDofeugh43Q6zebNm71uzz516pTVJxjO6bnqDJZzOnfuXPPBBx+Yffv2mc8//9z89re/Na1atTIbN240xgTHuTxXncFyLhvz07vnjPGvc0po8rEXXnjBdO/e3YSFhZlrr73W6/Zgf3d2r4zQ0FATHx9vxo4da3bs2GEdr62tNQsWLDAul8uEh4ebG2+80RQXF3t9R3V1tZk+fbqJjo42bdu2NWlpaebAgQOXuhQv77//vpFU7zVp0iRjTPPVdfToUTNhwgQTGRlpIiMjzYQJE0xFRcUlqvLn6zx16pQZNWqUueyyy0xoaKjp1q2bmTRpUr0aAqHOhmqUZFauXGn1CYZzeq46g+WcPvDAA9b/Zl522WVm+PDhVmAyJjjOpTE/X2ewnMvG1A1N/nROHcYYY39eCgAAoGViTRMAAIANhCYAAAAbCE0AAAA2EJoAAABsIDQBAADYQGgCAACwgdAEAABgA6EJAADABkITgIAxbNgwpaen++S3N2/eLIfDoWPHjvnk9wH4HqEJAOpoKJwNGTJEpaWlcjqdvhkUAJ8jNAFoMU6fPt3kz4aFhcnlcsnhcDTjiAAEEkITAL908uRJ/fM//7Pat2+vzp07a/HixV7HHQ6H3njjDa+2jh07atWqVZKk/fv3y+Fw6E9/+pOGDRumNm3aaM2aNTp69Kjuvfdede3aVe3atVNKSopee+016zvuv/9+FRQU6LnnnpPD4ZDD4dD+/fsbvDy3bt069evXT+Hh4erRo0e9Mfbo0UOZmZl64IEHFBkZqW7dumn58uXN+ncCcOkQmgD4pUcffVTvv/++1q9fr40bN2rz5s1yu93n/T2PP/64Zs6cqV27dmn06NH67rvvlJqaqv/+7/9WSUmJfv3rX2vixIn6+OOPJUnPPfecBg8erClTpqi0tFSlpaVKSEio971ut1vjxo3TPffco+LiYmVkZGj+/PlWaDtr8eLFGjBggLZv365p06bpoYce0hdffNGkvwkA3wrx9QAAoK4TJ05oxYoVeuWVVzRy5EhJ0urVq9W1a9fz/q709HSNHTvWq23OnDnWP8+YMUMbNmzQf/3Xf2nQoEFyOp0KCwtTu3bt5HK5Gv3e7OxsDR8+XPPnz5ck9enTRzt37tSzzz6r+++/3+p32223adq0aZJ+CHBLlizR5s2bdcUVV5x3LQB8i5kmAH5n7969qqmp0eDBg6226OhoXX755ef9XQMGDPB6f+bMGT399NPq37+/YmJi1L59e23cuFEHDhw4r+/dtWuXrr/+eq+266+/Xnv27NGZM2estv79+1v/7HA45HK5VF5eft51APA9ZpoA+B1jzDn7OByOev0aWugdERHh9X7x4sVasmSJcnJylJKSooiICKWnp6umpua8x1h3UXhD4w4NDa037tra2vP6LQD+gZkmAH6nd+/eCg0N1UcffWS1VVRU6Msvv7TeX3bZZSotLbXe79mzR6dOnTrnd3/44Ye68847dd999+mqq65Sz549tWfPHq8+YWFhXrNFDenbt6+2bNni1VZYWKg+ffqodevW5xwHgMDDTBMAv9O+fXtNnjxZjz76qGJiYhQXF6d58+apVasf/3/ezTffrKVLl+q6665TbW2tHn/88XqzOg3p3bu31q1bp8LCQkVFRSk7O1tlZWW68sorrT49evTQxx9/rP3796t9+/aKjo6u9z2zZ8/WwIED9a//+q+6++67tXXrVi1dulTLli1rnj8CAL/DTBMAv/Tss8/qxhtv1JgxYzRixAjdcMMNSk1NtY4vXrxYCQkJuvHGGzV+/HjNmTNH7dq1O+f3zp8/X9dee61Gjx6tYcOGyeVy6a677vLqM2fOHLVu3Vp9+/bVZZdd1uB6p2uvvVZ/+tOflJeXp+TkZD355JNauHCh1yJwAMHFYewsHgAAAGjhmGkCAACwgdAEAABgA6EJAADABkITAACADYQmAAAAGwhNAAAANhCaAAAAbCA0AQAA2EBoAgAAsIHQBAAAYAOhCQAAwIb/DzahOEr7S8d8AAAAAElFTkSuQmCC",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.histplot(bank.duration, bins=100)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {
    "id": "Kwh38FtibKUL",
    "tags": []
   },
   "outputs": [],
   "source": [
    "# Make a copy for parsing\n",
    "bank_data = bank.copy()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "QSDkyFARbaHa",
    "outputId": "1806da00-4ea8-4539-92b9-5d6d4f56f38a",
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "management      :  1301\n",
      "blue-collar     :   708\n",
      "technician      :   840\n",
      "admin.          :   631\n",
      "services        :   369\n",
      "retired         :   516\n",
      "self-employed   :   187\n",
      "student         :   269\n",
      "unemployed      :   202\n",
      "entrepreneur    :   123\n",
      "housemaid       :   109\n",
      "unknown         :    34\n"
     ]
    }
   ],
   "source": [
    "# Explore People who made a deposit Vs Job category\n",
    "jobs = ['management','blue-collar','technician','admin.','services','retired','self-employed','student',\\\n",
    "        'unemployed','entrepreneur','housemaid','unknown']\n",
    "\n",
    "for j in jobs:\n",
    "    print(\"{:15} : {:5}\". format(j, len(bank_data[(bank_data.deposit == \"yes\") & (bank_data.job ==j)])))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "y0bp4tNxbbDK",
    "outputId": "f3c4d723-ebca-4982-e512-02b332ae1044",
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "job\n",
       "management       2566\n",
       "blue-collar      1944\n",
       "technician       1823\n",
       "admin.           1334\n",
       "services          923\n",
       "retired           778\n",
       "self-employed     405\n",
       "student           360\n",
       "unemployed        357\n",
       "entrepreneur      328\n",
       "housemaid         274\n",
       "unknown            70\n",
       "Name: count, dtype: int64"
      ]
     },
     "execution_count": 11,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Different types of job categories and their counts\n",
    "bank_data.job.value_counts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {
    "id": "Ac4JC2cGbeOS",
    "tags": []
   },
   "outputs": [],
   "source": [
    "# Combine similar jobs into categiroes\n",
    "bank_data['job'] = bank_data['job'].replace(['management', 'admin.'], 'white-collar')\n",
    "bank_data['job'] = bank_data['job'].replace(['services','housemaid'], 'pink-collar')\n",
    "bank_data['job'] = bank_data['job'].replace(['retired', 'student', 'unemployed', 'unknown'], 'other')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "xGBXD_D1bjRS",
    "outputId": "ecc423bf-adb9-4661-9b7f-bdeca4772354",
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "job\n",
       "white-collar     3900\n",
       "blue-collar      1944\n",
       "technician       1823\n",
       "other            1565\n",
       "pink-collar      1197\n",
       "self-employed     405\n",
       "entrepreneur      328\n",
       "Name: count, dtype: int64"
      ]
     },
     "execution_count": 13,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# New value counts\n",
    "bank_data.job.value_counts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "y-TxFTz6bnnp",
    "outputId": "d8cb6c0c-be58-43ac-d460-45bca255d194",
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "poutcome\n",
       "unknown    8326\n",
       "failure    1228\n",
       "success    1071\n",
       "other       537\n",
       "Name: count, dtype: int64"
      ]
     },
     "execution_count": 14,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "bank_data.poutcome.value_counts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "mClGC8b1bqZm",
    "outputId": "e10048c8-bab3-4070-fd84-72d7bc571b9d",
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "poutcome\n",
       "unknown    8863\n",
       "failure    1228\n",
       "success    1071\n",
       "Name: count, dtype: int64"
      ]
     },
     "execution_count": 15,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Combine 'unknown' and 'other' as 'other' isn't really match with either 'success' or 'failure'\n",
    "bank_data['poutcome'] = bank_data['poutcome'].replace(['other'] , 'unknown')\n",
    "bank_data.poutcome.value_counts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {
    "id": "GUWOnsBpbqgu",
    "tags": []
   },
   "outputs": [],
   "source": [
    "# Drop 'contact', as every participant has been contacted.\n",
    "bank_data.drop('contact', axis=1, inplace=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {
    "id": "H4uWM5tabqlc",
    "tags": []
   },
   "outputs": [],
   "source": [
    "# values for \"default\" : yes/no\n",
    "bank_data[\"default\"]\n",
    "bank_data['default_cat'] = bank_data['default'].map( {'yes':1, 'no':0} )\n",
    "bank_data.drop('default', axis=1,inplace = True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {
    "id": "rWTG22sKbqpF",
    "tags": []
   },
   "outputs": [],
   "source": [
    "# values for \"housing\" : yes/no\n",
    "bank_data[\"housing_cat\"]=bank_data['housing'].map({'yes':1, 'no':0})\n",
    "bank_data.drop('housing', axis=1,inplace = True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {
    "id": "KXs_4qdFbz19",
    "tags": []
   },
   "outputs": [],
   "source": [
    "# values for \"loan\" : yes/no\n",
    "bank_data[\"loan_cat\"] = bank_data['loan'].map({'yes':1, 'no':0})\n",
    "bank_data.drop('loan', axis=1, inplace=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {
    "id": "q7ZTyBMTbz6Z",
    "tags": []
   },
   "outputs": [],
   "source": [
    "# day  : last contact day of the month\n",
    "# month: last contact month of year\n",
    "# Drop 'month' and 'day' as they don't have any intrinsic meaning\n",
    "bank_data.drop('month', axis=1, inplace=True)\n",
    "bank_data.drop('day', axis=1, inplace=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {
    "id": "0trv7GLUbz9U",
    "tags": []
   },
   "outputs": [],
   "source": [
    "# values for \"deposit\" : yes/no\n",
    "bank_data[\"deposit_cat\"] = bank_data['deposit'].map({'yes':1, 'no':0})\n",
    "bank_data.drop('deposit', axis=1, inplace=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "qCszsb0Zb0AF",
    "outputId": "5ac1a56c-e0fe-4501-dca3-f6b2d9d561a9",
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Customers that have not been contacted before: 8324\n",
      "Maximum values on padys    : 854\n"
     ]
    }
   ],
   "source": [
    "# pdays: number of days that passed by after the client was last contacted from a previous campaign\n",
    "#       -1 means client was not previously contacted\n",
    "\n",
    "print(\"Customers that have not been contacted before:\", len(bank_data[bank_data.pdays==-1]))\n",
    "print(\"Maximum values on padys    :\", bank_data['pdays'].max())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {
    "id": "e265SwJ3b0Ck",
    "tags": []
   },
   "outputs": [],
   "source": [
    "# Map padys=-1 into a large value (10000 is used) to indicate that it is so far in the past that it has no effect\n",
    "bank_data.loc[bank_data['pdays'] == -1, 'pdays'] = 10000"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "metadata": {
    "id": "mpOxV2Vyb0Gs",
    "tags": []
   },
   "outputs": [],
   "source": [
    "# Create a new column: recent_pdays\n",
    "bank_data['recent_pdays'] = np.where(bank_data['pdays'], 1/bank_data.pdays, 1/bank_data.pdays)\n",
    "\n",
    "# Drop 'pdays'\n",
    "bank_data.drop('pdays', axis=1, inplace = True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 261
    },
    "id": "l2TQOny4cDUJ",
    "outputId": "724c1c88-7c40-49fa-cc8f-fb8edde7112e",
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>age</th>\n",
       "      <th>job</th>\n",
       "      <th>marital</th>\n",
       "      <th>education</th>\n",
       "      <th>balance</th>\n",
       "      <th>duration</th>\n",
       "      <th>campaign</th>\n",
       "      <th>previous</th>\n",
       "      <th>poutcome</th>\n",
       "      <th>default_cat</th>\n",
       "      <th>housing_cat</th>\n",
       "      <th>loan_cat</th>\n",
       "      <th>deposit_cat</th>\n",
       "      <th>recent_pdays</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>11157</th>\n",
       "      <td>33</td>\n",
       "      <td>blue-collar</td>\n",
       "      <td>single</td>\n",
       "      <td>primary</td>\n",
       "      <td>1</td>\n",
       "      <td>257</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>unknown</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0.000100</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>11158</th>\n",
       "      <td>39</td>\n",
       "      <td>pink-collar</td>\n",
       "      <td>married</td>\n",
       "      <td>secondary</td>\n",
       "      <td>733</td>\n",
       "      <td>83</td>\n",
       "      <td>4</td>\n",
       "      <td>0</td>\n",
       "      <td>unknown</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0.000100</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>11159</th>\n",
       "      <td>32</td>\n",
       "      <td>technician</td>\n",
       "      <td>single</td>\n",
       "      <td>secondary</td>\n",
       "      <td>29</td>\n",
       "      <td>156</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>unknown</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0.000100</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>11160</th>\n",
       "      <td>43</td>\n",
       "      <td>technician</td>\n",
       "      <td>married</td>\n",
       "      <td>secondary</td>\n",
       "      <td>0</td>\n",
       "      <td>9</td>\n",
       "      <td>2</td>\n",
       "      <td>5</td>\n",
       "      <td>failure</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0.005814</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>11161</th>\n",
       "      <td>34</td>\n",
       "      <td>technician</td>\n",
       "      <td>married</td>\n",
       "      <td>secondary</td>\n",
       "      <td>0</td>\n",
       "      <td>628</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>unknown</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0.000100</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "       age          job  marital  education  balance  duration  campaign  \\\n",
       "11157   33  blue-collar   single    primary        1       257         1   \n",
       "11158   39  pink-collar  married  secondary      733        83         4   \n",
       "11159   32   technician   single  secondary       29       156         2   \n",
       "11160   43   technician  married  secondary        0         9         2   \n",
       "11161   34   technician  married  secondary        0       628         1   \n",
       "\n",
       "       previous poutcome  default_cat  housing_cat  loan_cat  deposit_cat  \\\n",
       "11157         0  unknown            0            1         0            0   \n",
       "11158         0  unknown            0            0         0            0   \n",
       "11159         0  unknown            0            0         0            0   \n",
       "11160         5  failure            0            0         1            0   \n",
       "11161         0  unknown            0            0         0            0   \n",
       "\n",
       "       recent_pdays  \n",
       "11157      0.000100  \n",
       "11158      0.000100  \n",
       "11159      0.000100  \n",
       "11160      0.005814  \n",
       "11161      0.000100  "
      ]
     },
     "execution_count": 25,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "bank_data.tail()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 255
    },
    "id": "yVPKMHMjcDXq",
    "outputId": "84b72c43-9da8-4735-a05a-880f59ad2863",
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>age</th>\n",
       "      <th>balance</th>\n",
       "      <th>duration</th>\n",
       "      <th>campaign</th>\n",
       "      <th>previous</th>\n",
       "      <th>default_cat</th>\n",
       "      <th>housing_cat</th>\n",
       "      <th>loan_cat</th>\n",
       "      <th>deposit_cat</th>\n",
       "      <th>recent_pdays</th>\n",
       "      <th>...</th>\n",
       "      <th>marital_divorced</th>\n",
       "      <th>marital_married</th>\n",
       "      <th>marital_single</th>\n",
       "      <th>education_primary</th>\n",
       "      <th>education_secondary</th>\n",
       "      <th>education_tertiary</th>\n",
       "      <th>education_unknown</th>\n",
       "      <th>poutcome_failure</th>\n",
       "      <th>poutcome_success</th>\n",
       "      <th>poutcome_unknown</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>59</td>\n",
       "      <td>2343</td>\n",
       "      <td>1042</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0.0001</td>\n",
       "      <td>...</td>\n",
       "      <td>False</td>\n",
       "      <td>True</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>True</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>True</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>56</td>\n",
       "      <td>45</td>\n",
       "      <td>1467</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0.0001</td>\n",
       "      <td>...</td>\n",
       "      <td>False</td>\n",
       "      <td>True</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>True</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>True</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>41</td>\n",
       "      <td>1270</td>\n",
       "      <td>1389</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0.0001</td>\n",
       "      <td>...</td>\n",
       "      <td>False</td>\n",
       "      <td>True</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>True</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>True</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>55</td>\n",
       "      <td>2476</td>\n",
       "      <td>579</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0.0001</td>\n",
       "      <td>...</td>\n",
       "      <td>False</td>\n",
       "      <td>True</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>True</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>True</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>54</td>\n",
       "      <td>184</td>\n",
       "      <td>673</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0.0001</td>\n",
       "      <td>...</td>\n",
       "      <td>False</td>\n",
       "      <td>True</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>True</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>True</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>5 rows × 27 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "   age  balance  duration  campaign  previous  default_cat  housing_cat  \\\n",
       "0   59     2343      1042         1         0            0            1   \n",
       "1   56       45      1467         1         0            0            0   \n",
       "2   41     1270      1389         1         0            0            1   \n",
       "3   55     2476       579         1         0            0            1   \n",
       "4   54      184       673         2         0            0            0   \n",
       "\n",
       "   loan_cat  deposit_cat  recent_pdays  ...  marital_divorced  \\\n",
       "0         0            1        0.0001  ...             False   \n",
       "1         0            1        0.0001  ...             False   \n",
       "2         0            1        0.0001  ...             False   \n",
       "3         0            1        0.0001  ...             False   \n",
       "4         0            1        0.0001  ...             False   \n",
       "\n",
       "   marital_married  marital_single  education_primary  education_secondary  \\\n",
       "0             True           False              False                 True   \n",
       "1             True           False              False                 True   \n",
       "2             True           False              False                 True   \n",
       "3             True           False              False                 True   \n",
       "4             True           False              False                False   \n",
       "\n",
       "   education_tertiary  education_unknown  poutcome_failure  poutcome_success  \\\n",
       "0               False              False             False             False   \n",
       "1               False              False             False             False   \n",
       "2               False              False             False             False   \n",
       "3               False              False             False             False   \n",
       "4                True              False             False             False   \n",
       "\n",
       "   poutcome_unknown  \n",
       "0              True  \n",
       "1              True  \n",
       "2              True  \n",
       "3              True  \n",
       "4              True  \n",
       "\n",
       "[5 rows x 27 columns]"
      ]
     },
     "execution_count": 26,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Convert categorical variables to dummies\n",
    "bank_with_dummies = pd.get_dummies(data=bank_data, columns = ['job', 'marital', 'education', 'poutcome'], \\\n",
    "                                   prefix = ['job', 'marital', 'education', 'poutcome'])\n",
    "bank_with_dummies.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "d2ZWHfY3cDah",
    "outputId": "74f5d749-eaca-40c2-ff56-930d6f6fcda9",
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(11162, 27)"
      ]
     },
     "execution_count": 27,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "bank_with_dummies.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 349
    },
    "id": "qlvr3vf3cDeK",
    "outputId": "b3dfdf6c-9461-4b89-9fbb-f24451b4be08",
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>age</th>\n",
       "      <th>balance</th>\n",
       "      <th>duration</th>\n",
       "      <th>campaign</th>\n",
       "      <th>previous</th>\n",
       "      <th>default_cat</th>\n",
       "      <th>housing_cat</th>\n",
       "      <th>loan_cat</th>\n",
       "      <th>deposit_cat</th>\n",
       "      <th>recent_pdays</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>count</th>\n",
       "      <td>11162.000000</td>\n",
       "      <td>11162.000000</td>\n",
       "      <td>11162.000000</td>\n",
       "      <td>11162.000000</td>\n",
       "      <td>11162.000000</td>\n",
       "      <td>11162.000000</td>\n",
       "      <td>11162.000000</td>\n",
       "      <td>11162.000000</td>\n",
       "      <td>11162.000000</td>\n",
       "      <td>11162.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>mean</th>\n",
       "      <td>41.231948</td>\n",
       "      <td>1528.538524</td>\n",
       "      <td>371.993818</td>\n",
       "      <td>2.508421</td>\n",
       "      <td>0.832557</td>\n",
       "      <td>0.015051</td>\n",
       "      <td>0.473123</td>\n",
       "      <td>0.130801</td>\n",
       "      <td>0.473840</td>\n",
       "      <td>0.003124</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>std</th>\n",
       "      <td>11.913369</td>\n",
       "      <td>3225.413326</td>\n",
       "      <td>347.128386</td>\n",
       "      <td>2.722077</td>\n",
       "      <td>2.292007</td>\n",
       "      <td>0.121761</td>\n",
       "      <td>0.499299</td>\n",
       "      <td>0.337198</td>\n",
       "      <td>0.499338</td>\n",
       "      <td>0.030686</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>min</th>\n",
       "      <td>18.000000</td>\n",
       "      <td>-6847.000000</td>\n",
       "      <td>2.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000100</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>25%</th>\n",
       "      <td>32.000000</td>\n",
       "      <td>122.000000</td>\n",
       "      <td>138.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000100</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>50%</th>\n",
       "      <td>39.000000</td>\n",
       "      <td>550.000000</td>\n",
       "      <td>255.000000</td>\n",
       "      <td>2.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000100</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>75%</th>\n",
       "      <td>49.000000</td>\n",
       "      <td>1708.000000</td>\n",
       "      <td>496.000000</td>\n",
       "      <td>3.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.001919</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>max</th>\n",
       "      <td>95.000000</td>\n",
       "      <td>81204.000000</td>\n",
       "      <td>3881.000000</td>\n",
       "      <td>63.000000</td>\n",
       "      <td>58.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>1.000000</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                age       balance      duration      campaign      previous  \\\n",
       "count  11162.000000  11162.000000  11162.000000  11162.000000  11162.000000   \n",
       "mean      41.231948   1528.538524    371.993818      2.508421      0.832557   \n",
       "std       11.913369   3225.413326    347.128386      2.722077      2.292007   \n",
       "min       18.000000  -6847.000000      2.000000      1.000000      0.000000   \n",
       "25%       32.000000    122.000000    138.000000      1.000000      0.000000   \n",
       "50%       39.000000    550.000000    255.000000      2.000000      0.000000   \n",
       "75%       49.000000   1708.000000    496.000000      3.000000      1.000000   \n",
       "max       95.000000  81204.000000   3881.000000     63.000000     58.000000   \n",
       "\n",
       "        default_cat   housing_cat      loan_cat   deposit_cat  recent_pdays  \n",
       "count  11162.000000  11162.000000  11162.000000  11162.000000  11162.000000  \n",
       "mean       0.015051      0.473123      0.130801      0.473840      0.003124  \n",
       "std        0.121761      0.499299      0.337198      0.499338      0.030686  \n",
       "min        0.000000      0.000000      0.000000      0.000000      0.000100  \n",
       "25%        0.000000      0.000000      0.000000      0.000000      0.000100  \n",
       "50%        0.000000      0.000000      0.000000      0.000000      0.000100  \n",
       "75%        0.000000      1.000000      0.000000      1.000000      0.001919  \n",
       "max        1.000000      1.000000      1.000000      1.000000      1.000000  "
      ]
     },
     "execution_count": 28,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "bank_with_dummies.describe()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 449
    },
    "id": "xzJugl58cPmL",
    "outputId": "aa5f7a32-adfe-4b85-cc98-0f986d879082",
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "image/png": "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+vRpdO3aFQDw97//HXPnzsWBAwcwdOhQPPnkk9i3bx+2bt1qfozTp08jLS0NpaWl+MUvfuGtSyEijbGliYiC2g8//IDZs2cjIyMDMTExSE9PBwBUVFSgtLQUQ4YMMQcmABg5cqTk54uKirBjxw5069bN/GfgwIHmxyaiwBHm7RMgIvKmadOmIS0tDW+88QZSU1PR3t6OrKwsNDc3QwgBnU4nOV7eON/e3o5p06bhueees3rsHj16uPXcicizGJqIKGjV1NTg6NGjeO2113DdddcBAAoKCsz7Bw4ciA0bNqCpqQl6vR4A8PXXX0seY9iwYXj//ffRt29fhIXxI5UokLF7joiCVmxsLOLj4/H666/j2LFj+OKLL/Dwww+b98+ePRvt7e249957cfToUWzduhUvvvgiAJhboObNm4eLFy9i1qxZKCwsRHl5ObZt24Z77rkHbW1tXrkuInIPhiYiClohISHYtGkTioqKkJWVhYceeggvvPCCeX9MTAw+/vhjlJSUYOjQoXj88cfx5JNPAoB5nFNqaip27dqFtrY2TJ48GVlZWVi4cCEMBgNCQvgRSxRIOHuOiMgJGzZswN133w2TyYSIiAhvnw4ReRA74ImI7Fi/fj0yMjLQs2dPfPPNN1i6dClmzpzJwEQUhBiaiIjsqKysxJNPPonKykr06NEDt912G5YvX+7t0yIiL2D3HBEREZEKHKVIREREpAJDExEREZEKDE1EREREKjA0EREREanA0ERERESkAkMTERERkQoMTUREREQqMDQRERERqfD/AVP3VAISWrA2AAAAAElFTkSuQmCC",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Scatterplot showing age and balance\n",
    "bank_with_dummies.plot(kind='scatter', x='age', y='balance');\n",
    "\n",
    "# Across all ages, majority of people have savings of less than 20000."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 430
    },
    "id": "XarxfBihcPpU",
    "outputId": "c58f9b8d-af0a-41fd-fac7-0c237f296945",
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "image/png": "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1p133qnXX39dp06d8vWcAAAAAk6rQtO//vUv3XzzzZo1a5acTqemTp2qbdu2+XpuAAAAAaNVoSkhIUG5ubn68ssvtXLlSlVWVur222/XjTfeqNzcXB07dszX8wQAAPCrS7oQPDg4WA888IDeeOMNLViwQJ9//rmysrLUo0cPPfLII6qoqPDVPAEAAPzqkkLThx9+qGnTpql79+7Kzc1VVlaWPv/8c33wwQf68ssvdd999/lqngAAAH4V3Jofys3N1cqVK7V3717dc889WrNmje655x5dc823GSwuLk7Lly/XD37wA59OFgAAwF9aFZqWLVumiRMn6rHHHpPT6Wyx5vrrr9eKFSsuaXIAAACBolWhad++fResCQ0N1YQJE1ozPAAAQMBp1TVNK1eu1J/+9Kdm+//0pz9p9erVlzwpAACAQNOq0DR//nx17dq12f7o6GhlZ2df8qQAAAACTatC04EDBxQXF9dsf8+ePXXw4MFLnhQAAECgaVVoio6O1scff9xs/7/+9S916dLlkicFAAAQaFoVmh588EHNmDFDGzduVGNjoxobG/XBBx9o5syZevDBB309RwAAAL9r1d1zL774og4cOKBhw4YpOPjbIZqamvTII49wTRMAAGiXWhWaQkND9cc//lEvvPCC/vWvf6lDhw5KTExUz549fT0/AACAgNCq0HRGnz591KdPH1/NBQAAIGC1KjQ1NjZq1apV+tvf/qaqqio1NTV5Hf/ggw98MjkAAIBA0arQNHPmTK1atUpjxoxRQkKCHA6Hr+cFAAAQUFoVmvLz8/XGG2/onnvu8fV8AAAAAlKrHjkQGhqq3r17+3ouAAAAAatVoWnWrFl66aWXZJqmr+cDAAAQkFr18dzmzZu1ceNGvffee7rxxhsVEhLidfytt97yyeQAAAACRatCU8eOHfXAAw/4ei4AAAABq1WhaeXKlb6eBwAAQEBr1TVNkvTNN9/o/fff1/Lly1VTUyNJOnLkiGpra1s1Xk5OjhwOhzIzM619pmlq3rx5crlc6tChg4YOHardu3d7/Vx9fb2mT5+url27KiIiQmlpaTp8+LBXjdvtVkZGhgzDkGEYysjI0PHjx1s1TwAAcHVqVWg6cOCAEhMTdd999+mJJ57QsWPHJEkLFy5UVlbWRY+3fft2vfLKKxowYIDX/oULFyo3N1d5eXnavn27nE6n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",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "bank_with_dummies.plot(kind='hist', x='poutcome_success', y='duration');"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 349
    },
    "id": "h8AYAsvScPsI",
    "outputId": "4546a053-851a-491a-aa8a-4d1c06d91678",
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>age</th>\n",
       "      <th>balance</th>\n",
       "      <th>duration</th>\n",
       "      <th>campaign</th>\n",
       "      <th>previous</th>\n",
       "      <th>default_cat</th>\n",
       "      <th>housing_cat</th>\n",
       "      <th>loan_cat</th>\n",
       "      <th>deposit_cat</th>\n",
       "      <th>recent_pdays</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>count</th>\n",
       "      <td>5289.000000</td>\n",
       "      <td>5289.000000</td>\n",
       "      <td>5289.000000</td>\n",
       "      <td>5289.000000</td>\n",
       "      <td>5289.000000</td>\n",
       "      <td>5289.000000</td>\n",
       "      <td>5289.000000</td>\n",
       "      <td>5289.000000</td>\n",
       "      <td>5289.0</td>\n",
       "      <td>5289.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>mean</th>\n",
       "      <td>41.670070</td>\n",
       "      <td>1804.267915</td>\n",
       "      <td>537.294574</td>\n",
       "      <td>2.141047</td>\n",
       "      <td>1.170354</td>\n",
       "      <td>0.009832</td>\n",
       "      <td>0.365854</td>\n",
       "      <td>0.091511</td>\n",
       "      <td>1.0</td>\n",
       "      <td>0.004238</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>std</th>\n",
       "      <td>13.497781</td>\n",
       "      <td>3501.104777</td>\n",
       "      <td>392.525262</td>\n",
       "      <td>1.921826</td>\n",
       "      <td>2.553272</td>\n",
       "      <td>0.098676</td>\n",
       "      <td>0.481714</td>\n",
       "      <td>0.288361</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.035686</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>min</th>\n",
       "      <td>18.000000</td>\n",
       "      <td>-3058.000000</td>\n",
       "      <td>8.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>1.0</td>\n",
       "      <td>0.000100</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>25%</th>\n",
       "      <td>31.000000</td>\n",
       "      <td>210.000000</td>\n",
       "      <td>244.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>1.0</td>\n",
       "      <td>0.000100</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>50%</th>\n",
       "      <td>38.000000</td>\n",
       "      <td>733.000000</td>\n",
       "      <td>426.000000</td>\n",
       "      <td>2.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>1.0</td>\n",
       "      <td>0.000100</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>75%</th>\n",
       "      <td>50.000000</td>\n",
       "      <td>2159.000000</td>\n",
       "      <td>725.000000</td>\n",
       "      <td>3.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>1.0</td>\n",
       "      <td>0.005128</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>max</th>\n",
       "      <td>95.000000</td>\n",
       "      <td>81204.000000</td>\n",
       "      <td>3881.000000</td>\n",
       "      <td>32.000000</td>\n",
       "      <td>58.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>1.0</td>\n",
       "      <td>1.000000</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "               age       balance     duration     campaign     previous  \\\n",
       "count  5289.000000   5289.000000  5289.000000  5289.000000  5289.000000   \n",
       "mean     41.670070   1804.267915   537.294574     2.141047     1.170354   \n",
       "std      13.497781   3501.104777   392.525262     1.921826     2.553272   \n",
       "min      18.000000  -3058.000000     8.000000     1.000000     0.000000   \n",
       "25%      31.000000    210.000000   244.000000     1.000000     0.000000   \n",
       "50%      38.000000    733.000000   426.000000     2.000000     0.000000   \n",
       "75%      50.000000   2159.000000   725.000000     3.000000     1.000000   \n",
       "max      95.000000  81204.000000  3881.000000    32.000000    58.000000   \n",
       "\n",
       "       default_cat  housing_cat     loan_cat  deposit_cat  recent_pdays  \n",
       "count  5289.000000  5289.000000  5289.000000       5289.0   5289.000000  \n",
       "mean      0.009832     0.365854     0.091511          1.0      0.004238  \n",
       "std       0.098676     0.481714     0.288361          0.0      0.035686  \n",
       "min       0.000000     0.000000     0.000000          1.0      0.000100  \n",
       "25%       0.000000     0.000000     0.000000          1.0      0.000100  \n",
       "50%       0.000000     0.000000     0.000000          1.0      0.000100  \n",
       "75%       0.000000     1.000000     0.000000          1.0      0.005128  \n",
       "max       1.000000     1.000000     1.000000          1.0      1.000000  "
      ]
     },
     "execution_count": 31,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# People who sign up to a term deposite\n",
    "bank_with_dummies[bank_data.deposit_cat == 1].describe()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "Ch69k5NHcPuw",
    "outputId": "d18f1e1b-aafd-4c57-af72-30518ab5f7f1",
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "265"
      ]
     },
     "execution_count": 32,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# People signed up to a term deposite having a personal loan (loan_cat) and housing loan (housing_cat)\n",
    "len(bank_with_dummies[(bank_with_dummies.deposit_cat == 1) & (bank_with_dummies.loan_cat) & (bank_with_dummies.housing_cat)])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "O1Kleg3RcPy1",
    "outputId": "12428470-3c6e-4794-aac1-c4b82975e2b2",
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "52"
      ]
     },
     "execution_count": 33,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# People signed up to a term deposite with a credit default\n",
    "len(bank_with_dummies[(bank_with_dummies.deposit_cat == 1) & (bank_with_dummies.default_cat ==1)])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 559
    },
    "id": "YhoVN36jceEW",
    "outputId": "5e393995-abb3-4b91-e0d6-3e516123b7fa",
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<Axes: xlabel='job', ylabel='deposit_cat'>"
      ]
     },
     "execution_count": 34,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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mj9/vq6++mp5eXk5fXS4vMvNzS3yPXehwk6POXbsmGw2m9atW2dvY79WOowxevXVV3X11VfLx8dH7dq100cffSTpf6dMrVmzRhEREapevbo6d+6s7du3Szq3PBMnTtRPP/1k3w8uXLhQ0rnPzdmzZ+uOO+5QjRo19OKLL0o69z0oPDxc3t7euvrqqzVx4kSHdWyz2TRr1iz17dtXPj4+atasmT788EP78+dfHx988IG6d+8ub29vvfvuu5KkBQsWKDQ0VN7e3mrVqpVmzpxZYLyPP/64yP2rJCUnJ+vGG2+Uj4+PgoKCNHz4cP35558O9X366acO49SuXdu+3MXVd6k++ugjtW3bVj4+PvL391fPnj3ttRW37Bc6X9/hw4c1dOhQh212sf3796t///6qU6eO/P39dccdd9i/V0r/+y7w8ssvKzAwULVr17Zvz9GjR6tu3bpq3Lix5s+fX2D+77//vjp37ixvb2+1adPG4b1dmOXLl6tNmzby8vJS06ZNNW3aNPtzkyZNUtu2bQuMEx4erueee84+bLWOfvjhB3Xo0EHe3t6KiIjQli1biq2pxAxK7NixY6ZKlSpm8+bNxhhjEhISTEBAgLnuuuvsfVq2bGlmzZplFixYYKpVq2Z69uxpNm3aZFJSUkxoaKgZOHCgve+QIUPMHXfcYZ92ZGSkefjhh01GRobJyMgwubm55o8//jABAQFm3LhxJi0tzfz444+mV69epkePHsXWet9995mgoCDz8ccfm127dpmvvvrKvP/++8YYY37//XdTvXp1ExMTY9LS0swnn3xiAgICzPPPP28fv1u3bmbEiBH24eDgYPPaa6/Zh1977TXz9ddfm927d5s1a9aYkJAQ89hjj9mfP7/8nTt3Nhs3bjTbtm0zJ0+eLOkqv2wKW9+///675boeM2aMqVOnjlm4cKH57bffTFJSknnrrbeMMcbs2bPHSDKtWrUyn3/+udm+fbu55557THBwsDl79qwxxpT4deHsPLds2WKMMebMmTPmueeeMz/88IPZvXu3effdd0316tXNsmXLHKbv6+troqOjTVpamlmxYoWpXr26mTt37uVa3aWqW7dupmbNmmbEiBFm27Zt9mU8X//Fr1VJpnHjxua9994zO3fuNMOHDzc1a9Y0hw8fNsYYs3btWiPJHD161Bw7dsz87W9/Mz179iz29fr777+bunXrmrvuusts2rTJbN++3cyfP99s27bNGGPM9OnTja+vr1m6dKnZtm2bGTNmjKlWrZrZsWOHMabgdruwhvPTnzJlitmyZYvZtWuXef31142Hh4f57rvvCqyH0aNHm23btpm0tLTSWsVl3vDhw03Dhg3NypUrza+//mqGDBli6tSpY7Kysszy5cuNJLN9+3aTkZFhjh07Zow5t758fX3NhAkTzI4dO8yiRYuMzWYzX375pTHGmPz8fNOlSxdz2223mU2bNpkdO3aYkSNHGn9/f/tr5fnnnzc1atQwffr0MT/++KP56aefTH5+vtvWw5VSkvfcxa9tY4w5evSokWTWrl1rjDHs10rRP/7xD9OqVSuzatUqs2vXLrNgwQLj5eVl1q1bZ/9c6dSpk1m3bp359ddfTdeuXU3nzp2NMcacOnXKjBw50rRp08a+Hzx16pQx5tznZv369c28efPMrl27zN69e82qVauMr6+vWbhwodm1a5f58ssvTdOmTc2ECRPs9Ugy/v7+5q233jLbt283zzzzjPHw8DCpqanGmP+9Ppo2bWqWL19udu/ebfbv32/mzp1rGjRoYG9bvny5qVu3rlm4cKHDeMXtX3/++WdTs2ZN89prr5kdO3aYjRs3mg4dOpgHH3zQob5PPvnEYR36+fmZBQsWFFvfpfrjjz9M1apVzfTp082ePXvMzz//bN58801z4sQJp5d9y5YtJjc312RkZBhfX1+TkJDgsM0u9Oeff5oWLVqYoUOHmp9//tmkpqaagQMHmpCQEJOTk2OMOfddoFatWubxxx8327ZtM/PmzTOSTJ8+fcxLL71kduzYYV544QVTrVo1k56e7lBL48aNzUcffWRSU1PNsGHDTK1atUxWVpYxpuD+bPPmzaZKlSpm0qRJZvv27WbBggXGx8fHvs737dtnqlSpYn744Qd7/T/99JOx2Wxm165dxhhjuY5Onjxp6tWrZ/r372/+85//mBUrVpirr766wGfRpSA4uahjx45m6tSpxhhj+vXrZ1566SXj6elpsrOzTUZGhpFk0tLSzIIFC4wk89tvv9nHffPNN01gYKB9+OIvyBd/qBtjzLPPPmt69+7t0LZv3z77F4PCbN++3UgyiYmJhT7/j3/8w4SEhDjs8N98801Ts2ZNk5eXV2gtF+9gLvbBBx8Yf39/+/D55d+6dWuR47jbxctota6zs7ONl5eXPbRc7PwHyttvv21v+/XXX+2vCWNMiV8Xzs6zuA+GmJgYc/fddztMPzg42OTm5trb7r33XtO/f/8ip1GWdOvWzYSGhjq8fp9++mkTGhpqjCk8OD3zzDP24ZMnTxqbzWa++OILY8z/PuS3bdtm2rVrZ+666y77jqUo48aNM82aNTNnzpwp9PmGDRual156yaHtuuuuMzExMcYY6+BUmFtuucWMHDnSYT20b9++2DoropMnT5pq1aqZJUuW2NvOnDljGjZsaF599dUi12W3bt3M3/72N4e26667zjz99NPGGGPWrFljfH19zV9//eXQ55prrjFz5swxxpwLTtWqVTOHDh26DEtWdpXkPedMcGK/VjpOnjxpvL29TXJyskN7VFSUGTBggP298NVXX9mf+/e//20kmdOnTxtjzr2m27VrV2DakkxsbKxDW9euXc3LL7/s0PbOO++YBg0aOIwXHR3t0KdTp072AHr+9ZGQkODQJygoyLz33nsObS+88IKJjIx0GK+4/eugQYPMI4884jCNpKQkU6VKFfvyOhucLq7vUqWkpBhJZu/evQWec3bZL3xPXVhzYebNm1fg/ZCTk2N8fHzM6tWrjTH/+y5w/v1hjDEhISGma9eu9uHc3FxTo0YNs3TpUodaJk+ebO9z9uxZ07hxY/PKK68YYwruzwYOHGh69erlUN/o0aNN69at7cN9+/Z1+CdFbGys6d69u9PraM6cOaZu3brmzz//tD8/a9asUg1OnKrnou7du2vdunUyxigpKUl33HGHwsLCtGHDBq1du1aBgYFq1aqVJKl69eq65ppr7OM2aNBAhw4dKtH8UlJStHbtWtWsWdP+OD/9Xbt2acmSJQ7PJSUlaevWrfLw8FC3bt0KnWZaWpoiIyNls9nsbV26dNHJkyf1+++/O1XX2rVr1atXLzVq1Ei1atXS4MGDdfjwYYdD4p6enrr22mtLtLzuZLWu09LSlJOTo5tvvrnY6Vy4zA0aNJAkh+1ekteFs/O80OzZsxUREaF69eqpZs2aeuutt5Senu7Qp02bNvLw8HCqhrLohhtucHj9RkZGaufOncrLyyu0/4XbpEaNGqpVq1aB5e3Zs6euvvpqffDBB/L09LS39+3b1/56aNOmjaRzp+l27dpV1apVKzCv7Oxs/fHHH+rSpYtDe5cuXZSWlubU8uXl5emll17StddeK39/f9WsWVNffvllge0YERHh1PQqkl27duns2bMO67datWq6/vrrLdfvxZ9HF77uU1JSdPLkSfv6Pv/Ys2eP/RRMSQoODla9evVKcYnKh5K+54rDfq10pKam6q+//lKvXr0c1tfixYsdXrNW+6SiXPz5kpKSokmTJjnM6+GHH1ZGRoZOnTpl7xcZGekwXmRkZIH35oXTzszM1L59+xQVFeUw7RdffNFhOayWJSUlRQsXLnSYRp8+fZSfn689e/ZYLm9xy36p2rVrp5tvvllt27bVvffeq7feektHjx4t0bIXJTo62mFc6dy6+O2331SrVi17e926dfXXX385TLdNmzaqUuV/kSAwMNDhtDkPDw/5+/sXeL1cuI2rVq2qiIiIIj9/09LSCt0fXvj58fDDD2vp0qX666+/dPbsWS1ZskRDhw6V5NzrIy0tTe3atXP4ffPFr8NLxS/JXdS9e3fNmzdPP/30k6pUqaLWrVurW7duWr9+vY4ePerwoX7xlyqbzVbi8+Hz8/N122236ZVXXinwXIMGDZSfn69OnTrZ2xo1aqSvvvqq2GkaYxx2Lufbztdo5b///a9uueUWRUdH64UXXlDdunW1YcMGRUVFOfxOxsfHx6nplRVW63r37t1OTefC7X5++S/8AXFJXhc+Pj5OzfO8Dz74QE899ZSmTZumyMhI1apVS1OmTNH3339fZI3na6jIP3J2ZnlvvfVWLV++XKmpqQ47jrffflunT592mI4z26Ww95iz74dp06bptddeU0JCgv03F7GxsQUuAFEZrx5Y1GeVM+u3uNdBfn6+GjRoUOi5+hf++LoyrvOSOP8l7MLPtIt/P8l+rXScf+3++9//VqNGjRye8/Lysn+ptNonFeXi13p+fr4mTpyou+66q0Bfb2/vYqd18Tq7cNrna3nrrbcctrskh3/wScUvS35+vh599FENHz68wPybNGliH+fi/W1hv+8t7fe5h4eHEhMTlZycrC+//FJvvPGGxo8frxUrVkhybtmLMmnSJI0aNcqhLT8/X+Hh4VqyZEmB/hf+46ewz0RXvx8U9b4o7r153m233SYvLy998skn8vLyUk5Oju6++277skjFr6OSfrd2BcHJRTfeeKNOnDihhIQEdevWTTabTd26dVN8fLyOHj2qESNGuDxtT0/PAv+969ixo5YvX66mTZsWeeW0WrVqOQy3bdtW+fn5Wr9+vXr27Fmgf+vWrbV8+XKHF3NycrJq1apV4MO3MJs3b1Zubq6mTZtm30l+8MEHTi1jWXLx+rZa1y1atJCPj4/WrFmjYcOGXZEaSzrPpKQkde7cWTExMfY2Z/9rVZ589913BYZbtGjh9I6mMJMnT1bNmjV18803a926dWrdurUkFfqeuPbaa7Vo0SKdPXu2wE7G19dXDRs21IYNG3TjjTfa25OTk3X99dc7Vcv5o9n/93//J+ncjmPnzp0KDQ11dfEqjObNm8vT01MbNmzQwIEDJZ374rN582bFxsbajxaW9EhIx44ddeDAAVWtWlVNmzYt7bLLPWffc+e/lGVkZKhDhw6SVOBS1+zXSsf5C1Skp6cXeiTOmc/+wr53FKVjx47avn27mjdvXmy/7777ToMHD3YYPv9aKExgYKAaNWqk3bt364EHHnCqlqLq+/XXX4utr169eg4XY9q5c6fD0bLLyWazqUuXLurSpYuee+45BQcHa+PGjZe87PXr11f9+vUd2jp27Khly5bZL3RT2r777jv7/i03N1cpKSl64oknCu3bunVrbdiwwaEtOTlZLVu2tH9+VK1aVUOGDNGCBQvk5eWl+++/3370yJnXR+vWrfXOO+/o9OnT9n9sXvyZdak4Vc9Ffn5+at++vd599111795d0rkw9eOPP2rHjh32Nlc0bdpU33//vfbu3ausrCzl5+fr8ccf15EjRzRgwAD98MMP2r17t7788ksNHTq0yA+7pk2basiQIRo6dKg+/fRT7dmzR+vWrbPvBGJiYrRv3z49+eST2rZtm/71r3/p+eefV1xcnMMh26Jcc801ys3N1RtvvKHdu3frnXfe0ezZs11ebne5eH1brWtvb289/fTTGjNmjP1UiO+++87hqk6lraTzbN68uTZv3qzVq1drx44devbZZ7Vp06bLVp+77Nu3T3Fxcdq+fbuWLl2qN95445L+aXHe1KlT9cADD+imm27Stm3biuz3xBNPKDs7W/fff782b96snTt36p133rFfrWr06NF65ZVXtGzZMm3fvl1jx47V1q1bna6xefPm9v9OpqWl6dFHH9WBAwcuefkqgho1auixxx7T6NGjtWrVKqWmpurhhx/WqVOnFBUVpeDgYNlsNn3++efKzMx0uCpacXr27KnIyEj169dPq1ev1t69e5WcnKxnnnlGmzdvvsxLVfY5+57z8fHRDTfcoMmTJys1NVXffPONnnnmGYc+7NdKR61atTRq1Cg99dRTWrRokXbt2qUtW7bozTff1KJFi5yaRtOmTbVnzx5t3bpVWVlZysnJKbLvc889p8WLF2vChAn69ddflZaWpmXLlhXYvh9++KHmz5+vHTt26Pnnn9cPP/xQ5Jfq8yZMmKD4+HjNmDFDO3bs0C+//KIFCxZo+vTpTi2HJD399NP69ttv9fjjj2vr1q3auXOnPvvsMz355JP2PjfddJP++c9/6scff9TmzZsVHR1d6CnXpe3777/Xyy+/rM2bNys9PV0ff/yxMjMzFRoaWirLfrEHHnhAAQEBuuOOO5SUlKQ9e/Zo/fr1GjFihNOnrhbnzTff1CeffKJt27bp8ccf19GjR+2n1l1s5MiRWrNmjV544QXt2LFDixYt0j//+c8CR8mGDRumr7/+Wl988UWBaVmto4EDB6pKlSqKiopSamqqVq5cqalTp17ycl6I4HQJevTooby8PHtIqlOnjlq3bq169epd0n+ER40aJQ8PD/u00tPT1bBhQ23cuFF5eXnq06ePwsLCNGLECPn5+RW7M5g1a5buuecexcTEqFWrVnr44Yft52k3atRIK1eu1A8//KB27dopOjpaUVFRBT78itK+fXtNnz5dr7zyisLCwrRkyRLFx8e7vNzucvH6PnPmjOW6fvbZZzVy5Eg999xzCg0NVf/+/S/7b4NKMs/o6Gjddddd6t+/vzp16qTDhw87HH2qKAYPHqzTp0/r+uuv1+OPP64nn3xSjzzySKlM+7XXXtN9992nm266STt27Ci0j7+/v77++mudPHlS3bp1U3h4uN566y37Dnj48OEaOXKkRo4cqbZt22rVqlX67LPP1KJFC6dqePbZZ9WxY0f16dNH3bt311VXXVXm7g7vTpMnT9bdd9+tQYMGqWPHjvrtt9+0evVq1alTR40aNdLEiRM1duxYBQYGWn5hO89ms2nlypW68cYbNXToULVs2VL333+/9u7dq8DAwMu8RGVfSd5z8+fP19mzZxUREaERI0bYL2V9Hvu10vPCCy/oueeeU3x8vEJDQ9WnTx+tWLFCzZo1c2r8u+++W3//+9/Vo0cP1atXr8BtDy7Up08fff7550pMTNR1112nG264QdOnT1dwcLBDv4kTJ+r999+3H5lfsmSJ/Qh+UYYNG6a3335bCxcuVNu2bdWtWzctXLjQ6eWQzp0JsH79eu3cuVNdu3ZVhw4d9Oyzz9p/CyWdOw06KChIN954owYOHKhRo0Zdkfv++fr66ptvvtEtt9yili1b6plnntG0adPUt2/fUln2i1WvXl3ffPONmjRporvuukuhoaEaOnSoTp8+XSpHoCZPnqxXXnlF7dq1U1JSkv71r38pICCg0L4dO3bUBx98oPfff19hYWF67rnnNGnSJIdbRUjnzrDp3LmzQkJCCpySZ7WOatasqRUrVig1NVUdOnTQ+PHjCz0V+FLYzJU4IRAASln37t3Vvn17h3s1AQDcz2az6ZNPPuEfPRXU3r171axZM23ZskXt27cv1WkbY9SqVSs9+uijiouLK9VplwZ+4wQAAADArQ4dOqR33nlH+/fv10MPPeTucgpFcAIAAADgVoGBgQoICNDcuXNVp04dd5dTKE7VAwAAAAALXBwCAAAAACwQnAAAAADAAsEJAAAAACwQnAAAAADAAsEJAFAhPfjgg07fR2bdunWy2Ww6duzYZa0JAFB+cTlyAECFNGPGDHHhWABAaSE4AQAqJD8/P3eXAACoQDhVDwBQIV14ql5OTo6GDx+u+vXry9vbW3/729+0adOmAuNs3LhR7dq1k7e3tzp16qRffvnlClcNACirCE4AgApvzJgxWr58uRYtWqQff/xRzZs3V58+fXTkyBGHfqNHj9bUqVO1adMm1a9fX7fffrvOnj3rpqoBAGUJwQkAUKH9+eefmjVrlqZMmaK+ffuqdevWeuutt+Tj46N58+Y59H3++efVq1cvtW3bVosWLdLBgwf1ySefuKlyAEBZQnACAFRou3bt0tmzZ9WlSxd7W7Vq1XT99dcrLS3NoW9kZKT977p16yokJKRAHwBA5URwAgBUaOevrGez2Qq0X9xWGGf6AAAqPoITAKBCa968uTw9PbVhwwZ729mzZ7V582aFhoY69P3uu+/sfx89elQ7duxQq1atrlitAICyi8uRAwAqtBo1auixxx7T6NGjVbduXTVp0kSvvvqqTp06paioKIe+kyZNkr+/vwIDAzV+/HgFBAQ4fRNdAEDFRnACAFR4kydPVn5+vgYNGqQTJ04oIiJCq1evVp06dQr0GzFihHbu3Kl27drps88+k6enp5uqBgCUJTbDbdUBABXQgAED5OHhoXfffdfdpQAAKgB+4wQAqFByc3OVmpqqb7/9Vm3atHF3OQCACoLgBACoUP7zn/8oIiJCbdq0UXR0tLvLAQBUEJyqBwAAAAAWOOIEAAAAABYITgAAAABggeAEAAAAABYITgAAAABggeAEAAAAABYITgAAAABggeAEAAAAABYITgAAAABggeAEAAAAABb+H8LzSds15lqBAAAAAElFTkSuQmCC",
      "text/plain": [
       "<Figure size 1000x600 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Bar chart of job Vs deposite\n",
    "plt.figure(figsize = (10,6))\n",
    "sns.barplot(x='job', y = 'deposit_cat', data = bank_data)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 559
    },
    "id": "qslpuXU3ceHb",
    "outputId": "37dab3dc-01ef-4b12-af2f-27e641fc64b0",
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<Axes: xlabel='poutcome', ylabel='duration'>"
      ]
     },
     "execution_count": 35,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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q6s1depURUZGytvbW9HR0Vq4cKG9ffv27brzzjsVEBAgf39/9e7d2+k9ZNGiRerUqZNatGih6667Tq+88oq9raamRhMmTFB4eLhatGihdu3aKT093d6elpamq6++Wt7e3oqIiNAjjzzS8N8cXFaEFK5IM2bMUHJysr7++mvdcccduv/++/Xjjz+eMa68vFwDBw5UTU2NPv30UwUFBdnbpk+frtTUVBUUFKhDhw667777dOLECUlSfn6+kpOTde+996qwsFBpaWl64oknlJWVJUlKTEzU9u3b9cMPP0iS8vLyFBISory8PEnSiRMntGHDBiUkJNivt2fPHr377rv64IMP9MEHHygvL0/PP//8pfoWAY3e3Llz9fTTT6tt27YqLi7Wli1bdPToUU2ePFlbtmzRp59+qmbNmmnYsGE6efJkg772p59+qp07dyonJ0cffPCBjh07pj59+qhly5b6/PPPtX79erVs2VK33367ampqGvS1gaasuLhY9913nx588EHt3LlTubm5Gj58uOr6Z1FHjRql7OxsvfTSS9q5c6f+/Oc/q2XLlpJ++mXIrbfeqhYtWmjdunXKz8/Xgw8+aP/bYMGCBZo+fbqee+457dy5UzNnztQTTzyhxYsXS5JeeuklrVq1Sm+++aZ27dqlZcuW2b+Eeeutt5SRkaFXX31Vu3fv1rvvvuv0C1lcoSzAzURFRVkZGRlO6+Li4qynnnrKsizLkmQ9/vjj9rbKykrL4XBYH3/8sWVZlvXZZ59Zkqz//u//tuLi4qzhw4db1dXV9vi9e/dakqzXX3/dXrd9+3ZLkrVz507Lsixr5MiR1oABA5zm8Oijj1rXX3+9ZVmWdfLkSSskJMR66623LMuyrC5duljp6elWmzZtLMuyrA0bNlienp7WkSNHLMuyrKeeesry9fW1KioqnPbXrVu3en+fAFhWRkaGFRUVdc7tpaWlliSrsLDQsqz///n/6quvLMv6//eLsrIyy7J++lmNi4s772uMHj3aCg0NdXpfWbhwodWxY0fr5MmT9rrq6mrLx8fH+uSTTy7qGAH8v/z8fEuStW/fvjO2JSQkWJMmTXJaN2TIEGv06NGWZVnWrl27LElWTk7OWfc9bdo0q3379lZNTc1Zt0dGRlrLly93WvfMM89YPXr0sCzLsiZOnGj17dvX6X3glFmzZlkdOnQ4575xZeKMFK5InTt3tr/28/OTv7+/SktLncb0799f11xzjd58882zfn7h3/cRHh4uSfY+du7cqV69ejmN79Wrl3bv3q3a2lo5HA7deuutys3N1eHDh7V9+3Y99NBDqq2ttX9DdtNNN9m/5ZJ+umTR39/f6TVPnzOAi7Nnzx6NHDlS11xzjQICAtS+fXtJ0v79+xv0dWJjY53eV/Lz8/Xtt9/K399fLVu2VMuWLRUUFKR//etfTpcFAbg4cXFx6tevn2JjY3XPPfdowYIFKisrq9NzCwoK5OHh4XS1yOnbe/furebNm5+x7eDBgyoqKtLYsWPtn/GWLVvq2WeftX/Gx4wZo4KCAnXs2FGPPPKI1qxZYz//nnvuUVVVla655hqNGzdOK1eutM904cpFSMHtNGvW7IxT9MePH3d6fPqbnMPhOOPSnTvvvFNffPGFduzYcdbX+fd9OBwOSbL3YVmWve6U0+eUmJio3NxcffHFF4qLi9NVV12lW2+9VXl5ecrNzVViYqLxnAFcnLvuukuHDh3SggULtHnzZm3evFmS6nx5XV3ef6SffoHz706ePKn4+HgVFBQ4Ld98841GjhxZz6MBcDoPDw/l5OTo448/1vXXX6/MzEx17NhRe/fuveDPr4+Pz3n3fb7tp/5/vWDBAqef8W3btmnTpk2SpJtuukl79+7VM888o6qqKiUnJ+uXv/ylpJ/uGLpr1y69/PLL8vHx0fjx43Xrrbee9f0FVw5CCm6ndevWKi4uth9XVFRo7969xvt5/vnnNXr0aPXr1++cMXUu119/vdavX++0bsOGDerQoYM8PDwk/f/npN566y07mhISErR27dozPh8F4NI7dOiQdu7cqccff1z9+vVTp06d6vyb6lNat26tkpISp3+MXehvU0k//QNq9+7datOmja699lqnJTAw0PRQAJyHw+FQr169NGPGDH311Vfy8vLSypUrz/j3Q21trbZt22Y/jo2N1cmTJ+3PM5+uc+fO+uKLL84aN6GhofrZz36mf/zjH2f8jJ868y1JAQEBGjFihBYsWKAVK1bo7bfftj/D7ePjo8GDB+ull15Sbm6uNm7cqMLCwob6tsAFuP053E7fvn2VlZWlu+66S61atdITTzxhx4up//zP/1Rtba369u2r3NxcXXfddXV63pQpU3TzzTfrmWee0YgRI7Rx40bNmzfP6e48MTExCg4O1l/+8he99957kn6KqylTpkiSbrnllnrNGUD9nLpb3muvvabw8HDt379ff/rTn4z2kZiYqIMHD+qFF17QL3/5S61evVoff/yxAgICzvu8+++/Xy+++KKGDBli3wBj//79euedd/Too4+qbdu2F3NoAP7P5s2b9emnn2rgwIFq06aNNm/erIMHD6pTp07y8/PT5MmT9eGHH+rnP/+5MjIynP7Ydrt27TR69Gg9+OCDeumllxQXF6d//vOfKi0tVXJysiZMmKDMzEzde++9mjZtmgIDA7Vp0yb94he/UMeOHZWWlqZHHnlEAQEBSkpKUnV1tbZu3aqysjJNnjxZGRkZCg8PV5cuXdSsWTP99a9/VVhYmK666iplZWWptrZW3bp1k6+vr5YuXSofHx9FRUW57puJi8YZKbidadOm6dZbb9WgQYN0xx13aOjQofr5z39e7/1lZGQoOTlZffv21TfffFOn59x000168803lZ2drZiYGD355JN6+umnnW6h6nA47LNOvXv3lvTTb7MCAwN14403XvAfXgAaVrNmzZSdna38/HzFxMToD3/4g1588UWjfXTq1EmvvPKKXn75ZcXFxelvf/ubUlNTL/g8X19fff7557r66qs1fPhwderUSQ8++KCqqqp4LwAaUEBAgD7//HPdcccd6tChgx5//HHNmjVLSUlJevDBBzV69GiNGjVKCQkJat++vfr06eP0/Pnz5+uXv/ylxo8fr+uuu07jxo3T0aNHJUnBwcFat26dKisrlZCQoPj4eC1YsMC+NP83v/mNXn/9dWVlZSk2NlYJCQnKysqyz0i1bNlS//Ef/6GuXbvq5ptv1r59+/TRRx+pWbNmuuqqq7RgwQL16tVLnTt31qeffqr3339fwcHBl/cbiAblsE6/mBQAAAAAcF6ckQIAAAAAQ4QUAAAAABgipAAAAADAECEFAAAAAIYIKQAAAAAwREgBAAAAgCFCCgAAAAAMEVIAAAAAYIiQAgA0GWPGjNHQoUNdPQ0AQCNASAEAAACAIUIKAOAWEhMTNWHCBE2YMEFXXXWVgoOD9fjjj8uyLElSWVmZRo0apVatWsnX11dJSUnavXu3/fy0tDR16dLFaZ9z5sxRu3bt7O2LFy/We++9J4fDIYfDodzcXEnSgQMHdO+99yooKEh+fn7q2rWrNm/ebO9n/vz5+vnPfy4vLy917NhRS5cudXodh8OhV199VYMGDZKvr686deqkjRs36ttvv1ViYqL8/PzUo0cP7dmzx+l577//vuLj49WiRQtdc801mjFjhk6cONFA31EAwKVESAEA3MbixYvl6empzZs366WXXlJGRoZef/11ST9dlrd161atWrVKGzdulGVZuuOOO3T8+PE67Ts1NVXJycm6/fbbVVxcrOLiYvXs2VOVlZVKSEjQd999p1WrVunvf/+7pk6dqpMnT0qSVq5cqUmTJmnKlCnatm2bfve73+mBBx7QZ5995rT/Z555RqNGjVJBQYGuu+46jRw5Ur/73e80bdo0bd26VZI0YcIEe/wnn3yiX/3qV3rkkUe0Y8cOvfrqq8rKytJzzz3XEN9KAMClZgEA4AYSEhKsTp06WSdPnrTX/fGPf7Q6depkffPNN5Yk68svv7S3/fDDD5aPj4/15ptvWpZlWU899ZQVFxfntM+MjAwrKirKfjx69GhryJAhTmNeffVVy9/f3zp06NBZ59WzZ09r3LhxTuvuuece64477rAfS7Ief/xx+/HGjRstSdbChQvtdW+88YbVokUL+3Hv3r2tmTNnOu136dKlVnh4+FnnAQBwL5yRAgC4je7du8vhcNiPe/Tood27d2vHjh3y9PRUt27d7G3BwcHq2LGjdu7ceVGvWVBQoBtvvFFBQUFn3b5z50716tXLaV2vXr3OeN3OnTvbX4eGhkqSYmNjndb961//UkVFhSQpPz9fTz/9tFq2bGkv48aNU3FxsY4dO3ZRxwQAuPQ8XT0BAADqy7IsO7yaNWtmf57qlLpc9ufj43PBMf8ed6e/7inNmzc/Y/zZ1p26ZPDkyZOaMWOGhg8ffsbrtWjR4oJzAgC4FmekAABuY9OmTWc8jo6O1vXXX68TJ0443QDi0KFD+uabb9SpUydJUuvWrVVSUuIUUwUFBU778/LyUm1trdO6zp07q6CgQD/++ONZ59SpUyetX7/ead2GDRvs162vm266Sbt27dK11157xtKsGf97BgB3xzs1AMBtFBUVafLkydq1a5feeOMNZWZmatKkSYqOjtaQIUM0btw4rV+/Xn//+9/1q1/9Sj/72c80ZMgQST/d9e/gwYN64YUXtGfPHr388sv6+OOPnfbfrl07ff3119q1a5d++OEHHT9+XPfdd5/CwsI0dOhQffnll/rHP/6ht99+Wxs3bpQkPfroo8rKytKf//xn7d69W7Nnz9Y777yj1NTUizrWJ598UkuWLFFaWpq2b9+unTt3asWKFXr88ccvar8AgMuDkAIAuI1Ro0apqqpKv/jFL/Twww9r4sSJ+u1vfytJWrRokeLj4zVo0CD16NFDlmXpo48+si+f69Spk1555RW9/PLLiouL09/+9rczYmfcuHHq2LGjunbtqtatW+vLL7+Ul5eX1qxZozZt2uiOO+5QbGysnn/+eXl4eEiShg4dqrlz5+rFF1/UDTfcoFdffVWLFi1SYmLiRR3rbbfdpg8++EA5OTm6+eab1b17d82ePVtRUVEXtV8AwOXhsE6/oBwAABdITExUly5dNGfOHFdPBQCAC+KMFAAAAAAYIqQAAAAAwBCX9gEAAACAIc5IAQAAAIAhQgoAAAAADBFSAAAAAGCIkAIAAAAAQ4QUAAAAABgipAAAAADAECEFAAAAAIYIKQAAAAAwREgBAAAAgKH/BSKgeW8WaJCjAAAAAElFTkSuQmCC",
      "text/plain": [
       "<Figure size 1000x600 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Bar chart of \"previous outcome\" Vs \"call duration\"\n",
    "\n",
    "plt.figure(figsize = (10,6))\n",
    "sns.barplot(x='poutcome', y = 'duration', data = bank_data)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "metadata": {
    "id": "aG6QmD6QceKE",
    "tags": []
   },
   "outputs": [],
   "source": [
    "# make a copy\n",
    "bankcl = bank_with_dummies"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 945
    },
    "id": "IL_SEK9kceMv",
    "outputId": "2d1d2246-1ad7-4a01-b6ec-b98dcbc65066",
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>age</th>\n",
       "      <th>balance</th>\n",
       "      <th>duration</th>\n",
       "      <th>campaign</th>\n",
       "      <th>previous</th>\n",
       "      <th>default_cat</th>\n",
       "      <th>housing_cat</th>\n",
       "      <th>loan_cat</th>\n",
       "      <th>deposit_cat</th>\n",
       "      <th>recent_pdays</th>\n",
       "      <th>...</th>\n",
       "      <th>marital_divorced</th>\n",
       "      <th>marital_married</th>\n",
       "      <th>marital_single</th>\n",
       "      <th>education_primary</th>\n",
       "      <th>education_secondary</th>\n",
       "      <th>education_tertiary</th>\n",
       "      <th>education_unknown</th>\n",
       "      <th>poutcome_failure</th>\n",
       "      <th>poutcome_success</th>\n",
       "      <th>poutcome_unknown</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>age</th>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.112300</td>\n",
       "      <td>0.000189</td>\n",
       "      <td>-0.005278</td>\n",
       "      <td>0.020169</td>\n",
       "      <td>-0.011425</td>\n",
       "      <td>-0.168700</td>\n",
       "      <td>-0.031418</td>\n",
       "      <td>0.034901</td>\n",
       "      <td>0.019102</td>\n",
       "      <td>...</td>\n",
       "      <td>0.186349</td>\n",
       "      <td>0.318436</td>\n",
       "      <td>-0.467799</td>\n",
       "      <td>0.231150</td>\n",
       "      <td>-0.094400</td>\n",
       "      <td>-0.101372</td>\n",
       "      <td>0.077761</td>\n",
       "      <td>-0.008071</td>\n",
       "      <td>0.062114</td>\n",
       "      <td>-0.038992</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>balance</th>\n",
       "      <td>0.112300</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.022436</td>\n",
       "      <td>-0.013894</td>\n",
       "      <td>0.030805</td>\n",
       "      <td>-0.060954</td>\n",
       "      <td>-0.077092</td>\n",
       "      <td>-0.084589</td>\n",
       "      <td>0.081129</td>\n",
       "      <td>-0.004379</td>\n",
       "      <td>...</td>\n",
       "      <td>-0.017586</td>\n",
       "      <td>0.025431</td>\n",
       "      <td>-0.014994</td>\n",
       "      <td>-0.000673</td>\n",
       "      <td>-0.070609</td>\n",
       "      <td>0.069128</td>\n",
       "      <td>0.014596</td>\n",
       "      <td>0.001695</td>\n",
       "      <td>0.045603</td>\n",
       "      <td>-0.034524</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>duration</th>\n",
       "      <td>0.000189</td>\n",
       "      <td>0.022436</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>-0.041557</td>\n",
       "      <td>-0.026716</td>\n",
       "      <td>-0.009760</td>\n",
       "      <td>0.035051</td>\n",
       "      <td>-0.001914</td>\n",
       "      <td>0.451919</td>\n",
       "      <td>-0.014868</td>\n",
       "      <td>...</td>\n",
       "      <td>0.021364</td>\n",
       "      <td>-0.036179</td>\n",
       "      <td>0.023847</td>\n",
       "      <td>0.013405</td>\n",
       "      <td>0.003820</td>\n",
       "      <td>-0.006813</td>\n",
       "      <td>-0.015887</td>\n",
       "      <td>-0.033966</td>\n",
       "      <td>-0.022578</td>\n",
       "      <td>0.042725</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>campaign</th>\n",
       "      <td>-0.005278</td>\n",
       "      <td>-0.013894</td>\n",
       "      <td>-0.041557</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>-0.049699</td>\n",
       "      <td>0.030975</td>\n",
       "      <td>0.006660</td>\n",
       "      <td>0.034722</td>\n",
       "      <td>-0.128081</td>\n",
       "      <td>-0.026296</td>\n",
       "      <td>...</td>\n",
       "      <td>-0.006828</td>\n",
       "      <td>0.047722</td>\n",
       "      <td>-0.046165</td>\n",
       "      <td>0.019915</td>\n",
       "      <td>-0.013834</td>\n",
       "      <td>-0.005427</td>\n",
       "      <td>0.012976</td>\n",
       "      <td>-0.080188</td>\n",
       "      <td>-0.091807</td>\n",
       "      <td>0.128907</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>previous</th>\n",
       "      <td>0.020169</td>\n",
       "      <td>0.030805</td>\n",
       "      <td>-0.026716</td>\n",
       "      <td>-0.049699</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>-0.035273</td>\n",
       "      <td>-0.000840</td>\n",
       "      <td>-0.022668</td>\n",
       "      <td>0.139867</td>\n",
       "      <td>0.122076</td>\n",
       "      <td>...</td>\n",
       "      <td>-0.026566</td>\n",
       "      <td>-0.005176</td>\n",
       "      <td>0.023817</td>\n",
       "      <td>-0.024852</td>\n",
       "      <td>-0.004620</td>\n",
       "      <td>0.028146</td>\n",
       "      <td>-0.011898</td>\n",
       "      <td>0.335870</td>\n",
       "      <td>0.325477</td>\n",
       "      <td>-0.496921</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>default_cat</th>\n",
       "      <td>-0.011425</td>\n",
       "      <td>-0.060954</td>\n",
       "      <td>-0.009760</td>\n",
       "      <td>0.030975</td>\n",
       "      <td>-0.035273</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.011076</td>\n",
       "      <td>0.076434</td>\n",
       "      <td>-0.040680</td>\n",
       "      <td>-0.011290</td>\n",
       "      <td>...</td>\n",
       "      <td>0.019633</td>\n",
       "      <td>-0.006819</td>\n",
       "      <td>-0.006255</td>\n",
       "      <td>0.013858</td>\n",
       "      <td>-0.000618</td>\n",
       "      <td>-0.011768</td>\n",
       "      <td>0.005421</td>\n",
       "      <td>-0.024650</td>\n",
       "      <td>-0.040272</td>\n",
       "      <td>0.048403</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>housing_cat</th>\n",
       "      <td>-0.168700</td>\n",
       "      <td>-0.077092</td>\n",
       "      <td>0.035051</td>\n",
       "      <td>0.006660</td>\n",
       "      <td>-0.000840</td>\n",
       "      <td>0.011076</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.076761</td>\n",
       "      <td>-0.203888</td>\n",
       "      <td>-0.029350</td>\n",
       "      <td>...</td>\n",
       "      <td>0.007430</td>\n",
       "      <td>0.036305</td>\n",
       "      <td>-0.043817</td>\n",
       "      <td>0.017002</td>\n",
       "      <td>0.118514</td>\n",
       "      <td>-0.114955</td>\n",
       "      <td>-0.053191</td>\n",
       "      <td>0.087741</td>\n",
       "      <td>-0.136299</td>\n",
       "      <td>0.031375</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>loan_cat</th>\n",
       "      <td>-0.031418</td>\n",
       "      <td>-0.084589</td>\n",
       "      <td>-0.001914</td>\n",
       "      <td>0.034722</td>\n",
       "      <td>-0.022668</td>\n",
       "      <td>0.076434</td>\n",
       "      <td>0.076761</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>-0.110580</td>\n",
       "      <td>-0.012697</td>\n",
       "      <td>...</td>\n",
       "      <td>0.026463</td>\n",
       "      <td>0.044148</td>\n",
       "      <td>-0.065288</td>\n",
       "      <td>0.006854</td>\n",
       "      <td>0.079583</td>\n",
       "      <td>-0.067513</td>\n",
       "      <td>-0.050249</td>\n",
       "      <td>0.006264</td>\n",
       "      <td>-0.080370</td>\n",
       "      <td>0.053686</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>deposit_cat</th>\n",
       "      <td>0.034901</td>\n",
       "      <td>0.081129</td>\n",
       "      <td>0.451919</td>\n",
       "      <td>-0.128081</td>\n",
       "      <td>0.139867</td>\n",
       "      <td>-0.040680</td>\n",
       "      <td>-0.203888</td>\n",
       "      <td>-0.110580</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.034457</td>\n",
       "      <td>...</td>\n",
       "      <td>0.005228</td>\n",
       "      <td>-0.092157</td>\n",
       "      <td>0.094632</td>\n",
       "      <td>-0.063002</td>\n",
       "      <td>-0.051952</td>\n",
       "      <td>0.094598</td>\n",
       "      <td>0.014355</td>\n",
       "      <td>0.020714</td>\n",
       "      <td>0.286642</td>\n",
       "      <td>-0.224785</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>recent_pdays</th>\n",
       "      <td>0.019102</td>\n",
       "      <td>-0.004379</td>\n",
       "      <td>-0.014868</td>\n",
       "      <td>-0.026296</td>\n",
       "      <td>0.122076</td>\n",
       "      <td>-0.011290</td>\n",
       "      <td>-0.029350</td>\n",
       "      <td>-0.012697</td>\n",
       "      <td>0.034457</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>...</td>\n",
       "      <td>-0.020253</td>\n",
       "      <td>0.009583</td>\n",
       "      <td>0.003736</td>\n",
       "      <td>-0.007034</td>\n",
       "      <td>-0.017129</td>\n",
       "      <td>0.017346</td>\n",
       "      <td>0.013590</td>\n",
       "      <td>0.051422</td>\n",
       "      <td>0.119598</td>\n",
       "      <td>-0.126890</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>job_blue-collar</th>\n",
       "      <td>-0.066567</td>\n",
       "      <td>-0.046220</td>\n",
       "      <td>0.029986</td>\n",
       "      <td>0.005522</td>\n",
       "      <td>-0.039939</td>\n",
       "      <td>0.022779</td>\n",
       "      <td>0.189848</td>\n",
       "      <td>0.057956</td>\n",
       "      <td>-0.100840</td>\n",
       "      <td>-0.018514</td>\n",
       "      <td>...</td>\n",
       "      <td>-0.056240</td>\n",
       "      <td>0.109188</td>\n",
       "      <td>-0.077645</td>\n",
       "      <td>0.299737</td>\n",
       "      <td>0.076687</td>\n",
       "      <td>-0.298548</td>\n",
       "      <td>-0.000640</td>\n",
       "      <td>-0.018022</td>\n",
       "      <td>-0.077422</td>\n",
       "      <td>0.070330</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>job_entrepreneur</th>\n",
       "      <td>0.024176</td>\n",
       "      <td>0.005039</td>\n",
       "      <td>-0.000908</td>\n",
       "      <td>0.013883</td>\n",
       "      <td>-0.022470</td>\n",
       "      <td>0.022060</td>\n",
       "      <td>0.011492</td>\n",
       "      <td>0.042631</td>\n",
       "      <td>-0.034443</td>\n",
       "      <td>0.006251</td>\n",
       "      <td>...</td>\n",
       "      <td>0.006638</td>\n",
       "      <td>0.050746</td>\n",
       "      <td>-0.058665</td>\n",
       "      <td>-0.004788</td>\n",
       "      <td>-0.021132</td>\n",
       "      <td>0.026612</td>\n",
       "      <td>-0.001555</td>\n",
       "      <td>-0.001840</td>\n",
       "      <td>-0.035072</td>\n",
       "      <td>0.026966</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>job_other</th>\n",
       "      <td>0.296418</td>\n",
       "      <td>0.050744</td>\n",
       "      <td>0.010680</td>\n",
       "      <td>-0.050212</td>\n",
       "      <td>0.031191</td>\n",
       "      <td>-0.018130</td>\n",
       "      <td>-0.233309</td>\n",
       "      <td>-0.096196</td>\n",
       "      <td>0.144408</td>\n",
       "      <td>0.024356</td>\n",
       "      <td>...</td>\n",
       "      <td>0.032824</td>\n",
       "      <td>-0.030982</td>\n",
       "      <td>0.010413</td>\n",
       "      <td>0.114003</td>\n",
       "      <td>-0.020532</td>\n",
       "      <td>-0.110383</td>\n",
       "      <td>0.112986</td>\n",
       "      <td>-0.010865</td>\n",
       "      <td>0.099733</td>\n",
       "      <td>-0.064228</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>job_pink-collar</th>\n",
       "      <td>-0.027942</td>\n",
       "      <td>-0.041063</td>\n",
       "      <td>0.005345</td>\n",
       "      <td>0.011958</td>\n",
       "      <td>-0.028623</td>\n",
       "      <td>-0.007173</td>\n",
       "      <td>0.043884</td>\n",
       "      <td>0.014969</td>\n",
       "      <td>-0.051717</td>\n",
       "      <td>-0.001183</td>\n",
       "      <td>...</td>\n",
       "      <td>0.025640</td>\n",
       "      <td>0.007558</td>\n",
       "      <td>-0.025718</td>\n",
       "      <td>0.056150</td>\n",
       "      <td>0.137129</td>\n",
       "      <td>-0.184418</td>\n",
       "      <td>-0.004629</td>\n",
       "      <td>-0.010816</td>\n",
       "      <td>-0.030331</td>\n",
       "      <td>0.030459</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>job_self-employed</th>\n",
       "      <td>-0.023163</td>\n",
       "      <td>0.020264</td>\n",
       "      <td>0.013506</td>\n",
       "      <td>0.001776</td>\n",
       "      <td>-0.002338</td>\n",
       "      <td>0.007493</td>\n",
       "      <td>-0.016903</td>\n",
       "      <td>0.004299</td>\n",
       "      <td>-0.004707</td>\n",
       "      <td>-0.008226</td>\n",
       "      <td>...</td>\n",
       "      <td>-0.011849</td>\n",
       "      <td>-0.008164</td>\n",
       "      <td>0.016864</td>\n",
       "      <td>-0.037121</td>\n",
       "      <td>-0.060080</td>\n",
       "      <td>0.097929</td>\n",
       "      <td>-0.016336</td>\n",
       "      <td>-0.010039</td>\n",
       "      <td>-0.001399</td>\n",
       "      <td>0.008786</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>job_technician</th>\n",
       "      <td>-0.082716</td>\n",
       "      <td>0.003802</td>\n",
       "      <td>-0.010440</td>\n",
       "      <td>0.021738</td>\n",
       "      <td>0.002035</td>\n",
       "      <td>0.003109</td>\n",
       "      <td>0.006551</td>\n",
       "      <td>0.006864</td>\n",
       "      <td>-0.011557</td>\n",
       "      <td>-0.007412</td>\n",
       "      <td>...</td>\n",
       "      <td>-0.005434</td>\n",
       "      <td>-0.052492</td>\n",
       "      <td>0.059696</td>\n",
       "      <td>-0.144948</td>\n",
       "      <td>0.152542</td>\n",
       "      <td>-0.041988</td>\n",
       "      <td>-0.034276</td>\n",
       "      <td>0.005763</td>\n",
       "      <td>-0.014744</td>\n",
       "      <td>0.006279</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>job_white-collar</th>\n",
       "      <td>-0.080122</td>\n",
       "      <td>0.013780</td>\n",
       "      <td>-0.031980</td>\n",
       "      <td>0.001944</td>\n",
       "      <td>0.034929</td>\n",
       "      <td>-0.013425</td>\n",
       "      <td>-0.012111</td>\n",
       "      <td>-0.007871</td>\n",
       "      <td>0.031621</td>\n",
       "      <td>0.004516</td>\n",
       "      <td>...</td>\n",
       "      <td>0.010701</td>\n",
       "      <td>-0.043270</td>\n",
       "      <td>0.038752</td>\n",
       "      <td>-0.229245</td>\n",
       "      <td>-0.222261</td>\n",
       "      <td>0.422261</td>\n",
       "      <td>-0.045233</td>\n",
       "      <td>0.029387</td>\n",
       "      <td>0.033044</td>\n",
       "      <td>-0.046804</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>marital_divorced</th>\n",
       "      <td>0.186349</td>\n",
       "      <td>-0.017586</td>\n",
       "      <td>0.021364</td>\n",
       "      <td>-0.006828</td>\n",
       "      <td>-0.026566</td>\n",
       "      <td>0.019633</td>\n",
       "      <td>0.007430</td>\n",
       "      <td>0.026463</td>\n",
       "      <td>0.005228</td>\n",
       "      <td>-0.020253</td>\n",
       "      <td>...</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>-0.415878</td>\n",
       "      <td>-0.245556</td>\n",
       "      <td>0.024821</td>\n",
       "      <td>0.009891</td>\n",
       "      <td>-0.024597</td>\n",
       "      <td>-0.008920</td>\n",
       "      <td>-0.026169</td>\n",
       "      <td>-0.018120</td>\n",
       "      <td>0.033445</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>marital_married</th>\n",
       "      <td>0.318436</td>\n",
       "      <td>0.025431</td>\n",
       "      <td>-0.036179</td>\n",
       "      <td>0.047722</td>\n",
       "      <td>-0.005176</td>\n",
       "      <td>-0.006819</td>\n",
       "      <td>0.036305</td>\n",
       "      <td>0.044148</td>\n",
       "      <td>-0.092157</td>\n",
       "      <td>0.009583</td>\n",
       "      <td>...</td>\n",
       "      <td>-0.415878</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>-0.779455</td>\n",
       "      <td>0.130232</td>\n",
       "      <td>0.001536</td>\n",
       "      <td>-0.098449</td>\n",
       "      <td>0.005451</td>\n",
       "      <td>0.007682</td>\n",
       "      <td>-0.010063</td>\n",
       "      <td>0.001384</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>marital_single</th>\n",
       "      <td>-0.467799</td>\n",
       "      <td>-0.014994</td>\n",
       "      <td>0.023847</td>\n",
       "      <td>-0.046165</td>\n",
       "      <td>0.023817</td>\n",
       "      <td>-0.006255</td>\n",
       "      <td>-0.043817</td>\n",
       "      <td>-0.065288</td>\n",
       "      <td>0.094632</td>\n",
       "      <td>0.003736</td>\n",
       "      <td>...</td>\n",
       "      <td>-0.245556</td>\n",
       "      <td>-0.779455</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>-0.155917</td>\n",
       "      <td>-0.008450</td>\n",
       "      <td>0.121884</td>\n",
       "      <td>0.000334</td>\n",
       "      <td>0.009838</td>\n",
       "      <td>0.023208</td>\n",
       "      <td>-0.024514</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>education_primary</th>\n",
       "      <td>0.231150</td>\n",
       "      <td>-0.000673</td>\n",
       "      <td>0.013405</td>\n",
       "      <td>0.019915</td>\n",
       "      <td>-0.024852</td>\n",
       "      <td>0.013858</td>\n",
       "      <td>0.017002</td>\n",
       "      <td>0.006854</td>\n",
       "      <td>-0.063002</td>\n",
       "      <td>-0.007034</td>\n",
       "      <td>...</td>\n",
       "      <td>0.024821</td>\n",
       "      <td>0.130232</td>\n",
       "      <td>-0.155917</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>-0.386670</td>\n",
       "      <td>-0.276834</td>\n",
       "      <td>-0.085057</td>\n",
       "      <td>-0.026044</td>\n",
       "      <td>-0.049879</td>\n",
       "      <td>0.056477</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>education_secondary</th>\n",
       "      <td>-0.094400</td>\n",
       "      <td>-0.070609</td>\n",
       "      <td>0.003820</td>\n",
       "      <td>-0.013834</td>\n",
       "      <td>-0.004620</td>\n",
       "      <td>-0.000618</td>\n",
       "      <td>0.118514</td>\n",
       "      <td>0.079583</td>\n",
       "      <td>-0.051952</td>\n",
       "      <td>-0.017129</td>\n",
       "      <td>...</td>\n",
       "      <td>0.009891</td>\n",
       "      <td>0.001536</td>\n",
       "      <td>-0.008450</td>\n",
       "      <td>-0.386670</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>-0.689501</td>\n",
       "      <td>-0.211849</td>\n",
       "      <td>0.010625</td>\n",
       "      <td>-0.029466</td>\n",
       "      <td>0.013238</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>education_tertiary</th>\n",
       "      <td>-0.101372</td>\n",
       "      <td>0.069128</td>\n",
       "      <td>-0.006813</td>\n",
       "      <td>-0.005427</td>\n",
       "      <td>0.028146</td>\n",
       "      <td>-0.011768</td>\n",
       "      <td>-0.114955</td>\n",
       "      <td>-0.067513</td>\n",
       "      <td>0.094598</td>\n",
       "      <td>0.017346</td>\n",
       "      <td>...</td>\n",
       "      <td>-0.024597</td>\n",
       "      <td>-0.098449</td>\n",
       "      <td>0.121884</td>\n",
       "      <td>-0.276834</td>\n",
       "      <td>-0.689501</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>-0.151672</td>\n",
       "      <td>0.012265</td>\n",
       "      <td>0.059518</td>\n",
       "      <td>-0.052836</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>education_unknown</th>\n",
       "      <td>0.077761</td>\n",
       "      <td>0.014596</td>\n",
       "      <td>-0.015887</td>\n",
       "      <td>0.012976</td>\n",
       "      <td>-0.011898</td>\n",
       "      <td>0.005421</td>\n",
       "      <td>-0.053191</td>\n",
       "      <td>-0.050249</td>\n",
       "      <td>0.014355</td>\n",
       "      <td>0.013590</td>\n",
       "      <td>...</td>\n",
       "      <td>-0.008920</td>\n",
       "      <td>0.005451</td>\n",
       "      <td>0.000334</td>\n",
       "      <td>-0.085057</td>\n",
       "      <td>-0.211849</td>\n",
       "      <td>-0.151672</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>-0.010658</td>\n",
       "      <td>0.018158</td>\n",
       "      <td>-0.004978</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>poutcome_failure</th>\n",
       "      <td>-0.008071</td>\n",
       "      <td>0.001695</td>\n",
       "      <td>-0.033966</td>\n",
       "      <td>-0.080188</td>\n",
       "      <td>0.335870</td>\n",
       "      <td>-0.024650</td>\n",
       "      <td>0.087741</td>\n",
       "      <td>0.006264</td>\n",
       "      <td>0.020714</td>\n",
       "      <td>0.051422</td>\n",
       "      <td>...</td>\n",
       "      <td>-0.026169</td>\n",
       "      <td>0.007682</td>\n",
       "      <td>0.009838</td>\n",
       "      <td>-0.026044</td>\n",
       "      <td>0.010625</td>\n",
       "      <td>0.012265</td>\n",
       "      <td>-0.010658</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>-0.114542</td>\n",
       "      <td>-0.690332</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>poutcome_success</th>\n",
       "      <td>0.062114</td>\n",
       "      <td>0.045603</td>\n",
       "      <td>-0.022578</td>\n",
       "      <td>-0.091807</td>\n",
       "      <td>0.325477</td>\n",
       "      <td>-0.040272</td>\n",
       "      <td>-0.136299</td>\n",
       "      <td>-0.080370</td>\n",
       "      <td>0.286642</td>\n",
       "      <td>0.119598</td>\n",
       "      <td>...</td>\n",
       "      <td>-0.018120</td>\n",
       "      <td>-0.010063</td>\n",
       "      <td>0.023208</td>\n",
       "      <td>-0.049879</td>\n",
       "      <td>-0.029466</td>\n",
       "      <td>0.059518</td>\n",
       "      <td>0.018158</td>\n",
       "      <td>-0.114542</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>-0.639659</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>poutcome_unknown</th>\n",
       "      <td>-0.038992</td>\n",
       "      <td>-0.034524</td>\n",
       "      <td>0.042725</td>\n",
       "      <td>0.128907</td>\n",
       "      <td>-0.496921</td>\n",
       "      <td>0.048403</td>\n",
       "      <td>0.031375</td>\n",
       "      <td>0.053686</td>\n",
       "      <td>-0.224785</td>\n",
       "      <td>-0.126890</td>\n",
       "      <td>...</td>\n",
       "      <td>0.033445</td>\n",
       "      <td>0.001384</td>\n",
       "      <td>-0.024514</td>\n",
       "      <td>0.056477</td>\n",
       "      <td>0.013238</td>\n",
       "      <td>-0.052836</td>\n",
       "      <td>-0.004978</td>\n",
       "      <td>-0.690332</td>\n",
       "      <td>-0.639659</td>\n",
       "      <td>1.000000</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>27 rows × 27 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "                          age   balance  duration  campaign  previous  \\\n",
       "age                  1.000000  0.112300  0.000189 -0.005278  0.020169   \n",
       "balance              0.112300  1.000000  0.022436 -0.013894  0.030805   \n",
       "duration             0.000189  0.022436  1.000000 -0.041557 -0.026716   \n",
       "campaign            -0.005278 -0.013894 -0.041557  1.000000 -0.049699   \n",
       "previous             0.020169  0.030805 -0.026716 -0.049699  1.000000   \n",
       "default_cat         -0.011425 -0.060954 -0.009760  0.030975 -0.035273   \n",
       "housing_cat         -0.168700 -0.077092  0.035051  0.006660 -0.000840   \n",
       "loan_cat            -0.031418 -0.084589 -0.001914  0.034722 -0.022668   \n",
       "deposit_cat          0.034901  0.081129  0.451919 -0.128081  0.139867   \n",
       "recent_pdays         0.019102 -0.004379 -0.014868 -0.026296  0.122076   \n",
       "job_blue-collar     -0.066567 -0.046220  0.029986  0.005522 -0.039939   \n",
       "job_entrepreneur     0.024176  0.005039 -0.000908  0.013883 -0.022470   \n",
       "job_other            0.296418  0.050744  0.010680 -0.050212  0.031191   \n",
       "job_pink-collar     -0.027942 -0.041063  0.005345  0.011958 -0.028623   \n",
       "job_self-employed   -0.023163  0.020264  0.013506  0.001776 -0.002338   \n",
       "job_technician      -0.082716  0.003802 -0.010440  0.021738  0.002035   \n",
       "job_white-collar    -0.080122  0.013780 -0.031980  0.001944  0.034929   \n",
       "marital_divorced     0.186349 -0.017586  0.021364 -0.006828 -0.026566   \n",
       "marital_married      0.318436  0.025431 -0.036179  0.047722 -0.005176   \n",
       "marital_single      -0.467799 -0.014994  0.023847 -0.046165  0.023817   \n",
       "education_primary    0.231150 -0.000673  0.013405  0.019915 -0.024852   \n",
       "education_secondary -0.094400 -0.070609  0.003820 -0.013834 -0.004620   \n",
       "education_tertiary  -0.101372  0.069128 -0.006813 -0.005427  0.028146   \n",
       "education_unknown    0.077761  0.014596 -0.015887  0.012976 -0.011898   \n",
       "poutcome_failure    -0.008071  0.001695 -0.033966 -0.080188  0.335870   \n",
       "poutcome_success     0.062114  0.045603 -0.022578 -0.091807  0.325477   \n",
       "poutcome_unknown    -0.038992 -0.034524  0.042725  0.128907 -0.496921   \n",
       "\n",
       "                     default_cat  housing_cat  loan_cat  deposit_cat  \\\n",
       "age                    -0.011425    -0.168700 -0.031418     0.034901   \n",
       "balance                -0.060954    -0.077092 -0.084589     0.081129   \n",
       "duration               -0.009760     0.035051 -0.001914     0.451919   \n",
       "campaign                0.030975     0.006660  0.034722    -0.128081   \n",
       "previous               -0.035273    -0.000840 -0.022668     0.139867   \n",
       "default_cat             1.000000     0.011076  0.076434    -0.040680   \n",
       "housing_cat             0.011076     1.000000  0.076761    -0.203888   \n",
       "loan_cat                0.076434     0.076761  1.000000    -0.110580   \n",
       "deposit_cat            -0.040680    -0.203888 -0.110580     1.000000   \n",
       "recent_pdays           -0.011290    -0.029350 -0.012697     0.034457   \n",
       "job_blue-collar         0.022779     0.189848  0.057956    -0.100840   \n",
       "job_entrepreneur        0.022060     0.011492  0.042631    -0.034443   \n",
       "job_other              -0.018130    -0.233309 -0.096196     0.144408   \n",
       "job_pink-collar        -0.007173     0.043884  0.014969    -0.051717   \n",
       "job_self-employed       0.007493    -0.016903  0.004299    -0.004707   \n",
       "job_technician          0.003109     0.006551  0.006864    -0.011557   \n",
       "job_white-collar       -0.013425    -0.012111 -0.007871     0.031621   \n",
       "marital_divorced        0.019633     0.007430  0.026463     0.005228   \n",
       "marital_married        -0.006819     0.036305  0.044148    -0.092157   \n",
       "marital_single         -0.006255    -0.043817 -0.065288     0.094632   \n",
       "education_primary       0.013858     0.017002  0.006854    -0.063002   \n",
       "education_secondary    -0.000618     0.118514  0.079583    -0.051952   \n",
       "education_tertiary     -0.011768    -0.114955 -0.067513     0.094598   \n",
       "education_unknown       0.005421    -0.053191 -0.050249     0.014355   \n",
       "poutcome_failure       -0.024650     0.087741  0.006264     0.020714   \n",
       "poutcome_success       -0.040272    -0.136299 -0.080370     0.286642   \n",
       "poutcome_unknown        0.048403     0.031375  0.053686    -0.224785   \n",
       "\n",
       "                     recent_pdays  ...  marital_divorced  marital_married  \\\n",
       "age                      0.019102  ...          0.186349         0.318436   \n",
       "balance                 -0.004379  ...         -0.017586         0.025431   \n",
       "duration                -0.014868  ...          0.021364        -0.036179   \n",
       "campaign                -0.026296  ...         -0.006828         0.047722   \n",
       "previous                 0.122076  ...         -0.026566        -0.005176   \n",
       "default_cat             -0.011290  ...          0.019633        -0.006819   \n",
       "housing_cat             -0.029350  ...          0.007430         0.036305   \n",
       "loan_cat                -0.012697  ...          0.026463         0.044148   \n",
       "deposit_cat              0.034457  ...          0.005228        -0.092157   \n",
       "recent_pdays             1.000000  ...         -0.020253         0.009583   \n",
       "job_blue-collar         -0.018514  ...         -0.056240         0.109188   \n",
       "job_entrepreneur         0.006251  ...          0.006638         0.050746   \n",
       "job_other                0.024356  ...          0.032824        -0.030982   \n",
       "job_pink-collar         -0.001183  ...          0.025640         0.007558   \n",
       "job_self-employed       -0.008226  ...         -0.011849        -0.008164   \n",
       "job_technician          -0.007412  ...         -0.005434        -0.052492   \n",
       "job_white-collar         0.004516  ...          0.010701        -0.043270   \n",
       "marital_divorced        -0.020253  ...          1.000000        -0.415878   \n",
       "marital_married          0.009583  ...         -0.415878         1.000000   \n",
       "marital_single           0.003736  ...         -0.245556        -0.779455   \n",
       "education_primary       -0.007034  ...          0.024821         0.130232   \n",
       "education_secondary     -0.017129  ...          0.009891         0.001536   \n",
       "education_tertiary       0.017346  ...         -0.024597        -0.098449   \n",
       "education_unknown        0.013590  ...         -0.008920         0.005451   \n",
       "poutcome_failure         0.051422  ...         -0.026169         0.007682   \n",
       "poutcome_success         0.119598  ...         -0.018120        -0.010063   \n",
       "poutcome_unknown        -0.126890  ...          0.033445         0.001384   \n",
       "\n",
       "                     marital_single  education_primary  education_secondary  \\\n",
       "age                       -0.467799           0.231150            -0.094400   \n",
       "balance                   -0.014994          -0.000673            -0.070609   \n",
       "duration                   0.023847           0.013405             0.003820   \n",
       "campaign                  -0.046165           0.019915            -0.013834   \n",
       "previous                   0.023817          -0.024852            -0.004620   \n",
       "default_cat               -0.006255           0.013858            -0.000618   \n",
       "housing_cat               -0.043817           0.017002             0.118514   \n",
       "loan_cat                  -0.065288           0.006854             0.079583   \n",
       "deposit_cat                0.094632          -0.063002            -0.051952   \n",
       "recent_pdays               0.003736          -0.007034            -0.017129   \n",
       "job_blue-collar           -0.077645           0.299737             0.076687   \n",
       "job_entrepreneur          -0.058665          -0.004788            -0.021132   \n",
       "job_other                  0.010413           0.114003            -0.020532   \n",
       "job_pink-collar           -0.025718           0.056150             0.137129   \n",
       "job_self-employed          0.016864          -0.037121            -0.060080   \n",
       "job_technician             0.059696          -0.144948             0.152542   \n",
       "job_white-collar           0.038752          -0.229245            -0.222261   \n",
       "marital_divorced          -0.245556           0.024821             0.009891   \n",
       "marital_married           -0.779455           0.130232             0.001536   \n",
       "marital_single             1.000000          -0.155917            -0.008450   \n",
       "education_primary         -0.155917           1.000000            -0.386670   \n",
       "education_secondary       -0.008450          -0.386670             1.000000   \n",
       "education_tertiary         0.121884          -0.276834            -0.689501   \n",
       "education_unknown          0.000334          -0.085057            -0.211849   \n",
       "poutcome_failure           0.009838          -0.026044             0.010625   \n",
       "poutcome_success           0.023208          -0.049879            -0.029466   \n",
       "poutcome_unknown          -0.024514           0.056477             0.013238   \n",
       "\n",
       "                     education_tertiary  education_unknown  poutcome_failure  \\\n",
       "age                           -0.101372           0.077761         -0.008071   \n",
       "balance                        0.069128           0.014596          0.001695   \n",
       "duration                      -0.006813          -0.015887         -0.033966   \n",
       "campaign                      -0.005427           0.012976         -0.080188   \n",
       "previous                       0.028146          -0.011898          0.335870   \n",
       "default_cat                   -0.011768           0.005421         -0.024650   \n",
       "housing_cat                   -0.114955          -0.053191          0.087741   \n",
       "loan_cat                      -0.067513          -0.050249          0.006264   \n",
       "deposit_cat                    0.094598           0.014355          0.020714   \n",
       "recent_pdays                   0.017346           0.013590          0.051422   \n",
       "job_blue-collar               -0.298548          -0.000640         -0.018022   \n",
       "job_entrepreneur               0.026612          -0.001555         -0.001840   \n",
       "job_other                     -0.110383           0.112986         -0.010865   \n",
       "job_pink-collar               -0.184418          -0.004629         -0.010816   \n",
       "job_self-employed              0.097929          -0.016336         -0.010039   \n",
       "job_technician                -0.041988          -0.034276          0.005763   \n",
       "job_white-collar               0.422261          -0.045233          0.029387   \n",
       "marital_divorced              -0.024597          -0.008920         -0.026169   \n",
       "marital_married               -0.098449           0.005451          0.007682   \n",
       "marital_single                 0.121884           0.000334          0.009838   \n",
       "education_primary             -0.276834          -0.085057         -0.026044   \n",
       "education_secondary           -0.689501          -0.211849          0.010625   \n",
       "education_tertiary             1.000000          -0.151672          0.012265   \n",
       "education_unknown             -0.151672           1.000000         -0.010658   \n",
       "poutcome_failure               0.012265          -0.010658          1.000000   \n",
       "poutcome_success               0.059518           0.018158         -0.114542   \n",
       "poutcome_unknown              -0.052836          -0.004978         -0.690332   \n",
       "\n",
       "                     poutcome_success  poutcome_unknown  \n",
       "age                          0.062114         -0.038992  \n",
       "balance                      0.045603         -0.034524  \n",
       "duration                    -0.022578          0.042725  \n",
       "campaign                    -0.091807          0.128907  \n",
       "previous                     0.325477         -0.496921  \n",
       "default_cat                 -0.040272          0.048403  \n",
       "housing_cat                 -0.136299          0.031375  \n",
       "loan_cat                    -0.080370          0.053686  \n",
       "deposit_cat                  0.286642         -0.224785  \n",
       "recent_pdays                 0.119598         -0.126890  \n",
       "job_blue-collar             -0.077422          0.070330  \n",
       "job_entrepreneur            -0.035072          0.026966  \n",
       "job_other                    0.099733         -0.064228  \n",
       "job_pink-collar             -0.030331          0.030459  \n",
       "job_self-employed           -0.001399          0.008786  \n",
       "job_technician              -0.014744          0.006279  \n",
       "job_white-collar             0.033044         -0.046804  \n",
       "marital_divorced            -0.018120          0.033445  \n",
       "marital_married             -0.010063          0.001384  \n",
       "marital_single               0.023208         -0.024514  \n",
       "education_primary           -0.049879          0.056477  \n",
       "education_secondary         -0.029466          0.013238  \n",
       "education_tertiary           0.059518         -0.052836  \n",
       "education_unknown            0.018158         -0.004978  \n",
       "poutcome_failure            -0.114542         -0.690332  \n",
       "poutcome_success             1.000000         -0.639659  \n",
       "poutcome_unknown            -0.639659          1.000000  \n",
       "\n",
       "[27 rows x 27 columns]"
      ]
     },
     "execution_count": 37,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# The Correltion matrix\n",
    "corr = bankcl.corr()\n",
    "corr"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 854
    },
    "id": "ZcX7MjytcePd",
    "outputId": "7a11a740-a97f-42cd-fea6-18b525172688",
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Text(0.5, 1.0, 'Heatmap of Correlation Matrix')"
      ]
     },
     "execution_count": 38,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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Dh48SEREEydOpFatWtGBAweoR48e1KpVK07zLJsS3K9fP2rdujUpKioyrUa4sGPHDqpXrx7jMlq3bl3atWsXZ91169aRv78/ERFduXKF1NXVmTTKknY3bOGz1y1fO998niMl5OTkkLu7OykqKpKioiLzvEyfPp2WL1/OSfv48eOkrq5O48aNI1VVVUZ7y5YtrLIkyva3/dqNLceOHSNzc3PatGkTRUZGyi0FmYho2rRppKOjQ3Z2djRhwgSxtkWenp6ctLdt20Y1a9YkLy8vOnToENNXs+TGlYKCAoqMjKTly5dTjx49SEdHh5SVlal169actceOHUtr167lrCMNTU1N5ntn0aJFjLv83bt3Obtla2trk4qKCrVt25bmz59PwcHB9N9//3GdssBPhhA8CvywtGzZkrm4Lk1wcDBj3X3y5ElWF9pqampS02WePXvGuvl7Caampoy1dmkCAgI428jPmzePJk+eLHbhX1hYSNOmTaNff/2VioqKaMKECZzTHkuTnJzM+Tnh8wLbxcWFduzYQUREc+fOJQsLC1q6dCnZ2dlRly5dOGnXrVuXCRhtbW2Z/m+RkZGc7dgPHDhAe/fuJSKie/fukYGBASkoKJCamhodOXKEk3ZZ5NWbLSoqiq5cuUJE/7/uTFtbm5o3b84EOWwZO3as2M3d3Z3mzZvHqR5MGm/fvpVLXVx5PH/+nP766y/OzwcRv71utbW1mc9AExMTZsHp6dOnnFpS8HmOlMBng/bSrRFKP+f3799ndeFeurftt25sKVkQKX2TV9sVPmvkpM279PzlxePHj2nbtm00aNAgUlJS4twrkYho6dKlpKenRwMHDqRly5aRv7+/2I0L+vr6zIJo+/btafv27URUfH3CtV1M2YBaW1ubCajnzZvHSVvg50EIHgV+WNTU1CghIUFiPCEhgTG5YPthamRkJLX59qVLl8jAwED2yZZCVVWVqekrTWJiIqmqqnLSrlGjhtTanidPnlD16tWJqNgISFdXl9NxSsjNzaWZM2dSw4YN5aJXgjwvsFNSUpjV9ZycHJo8eTLZ2NhQ//79OdcBDR8+nFldXrp0KRkYGNC4cePIxMSEs2FOWXJycuju3bv09u1buejx3ZtNgH/47HVbWTvffMB3g/aSoLq0dkpKCufPb75ITU396u1/kT///JOGDh1KNWvWJAMDAxowYAD5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",
      "text/plain": [
       "<Figure size 1000x1000 with 2 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Heatmap\n",
    "plt.figure(figsize = (10,10))\n",
    "cmap = sns.diverging_palette(220, 10, as_cmap=True)\n",
    "sns.heatmap(corr, xticklabels=corr.columns.values, yticklabels=corr.columns.values, cmap=cmap, vmax=.3, center=0, square=True, linewidths=.5, cbar_kws={\"shrink\": .82})\n",
    "plt.title('Heatmap of Correlation Matrix')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 864
    },
    "id": "IzkOVajEceSB",
    "outputId": "4621a35c-1c3d-4e0c-e203-fa642f202999",
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>deposit_cat</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>duration</th>\n",
       "      <td>0.451919</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>poutcome_success</th>\n",
       "      <td>0.286642</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>job_other</th>\n",
       "      <td>0.144408</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>previous</th>\n",
       "      <td>0.139867</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>marital_single</th>\n",
       "      <td>0.094632</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>education_tertiary</th>\n",
       "      <td>0.094598</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>balance</th>\n",
       "      <td>0.081129</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>age</th>\n",
       "      <td>0.034901</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>recent_pdays</th>\n",
       "      <td>0.034457</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>job_white-collar</th>\n",
       "      <td>0.031621</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>poutcome_failure</th>\n",
       "      <td>0.020714</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>education_unknown</th>\n",
       "      <td>0.014355</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>marital_divorced</th>\n",
       "      <td>0.005228</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>job_self-employed</th>\n",
       "      <td>-0.004707</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>job_technician</th>\n",
       "      <td>-0.011557</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>job_entrepreneur</th>\n",
       "      <td>-0.034443</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>default_cat</th>\n",
       "      <td>-0.040680</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>job_pink-collar</th>\n",
       "      <td>-0.051717</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>education_secondary</th>\n",
       "      <td>-0.051952</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>education_primary</th>\n",
       "      <td>-0.063002</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>marital_married</th>\n",
       "      <td>-0.092157</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>job_blue-collar</th>\n",
       "      <td>-0.100840</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>loan_cat</th>\n",
       "      <td>-0.110580</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>campaign</th>\n",
       "      <td>-0.128081</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>housing_cat</th>\n",
       "      <td>-0.203888</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>poutcome_unknown</th>\n",
       "      <td>-0.224785</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                     deposit_cat\n",
       "duration                0.451919\n",
       "poutcome_success        0.286642\n",
       "job_other               0.144408\n",
       "previous                0.139867\n",
       "marital_single          0.094632\n",
       "education_tertiary      0.094598\n",
       "balance                 0.081129\n",
       "age                     0.034901\n",
       "recent_pdays            0.034457\n",
       "job_white-collar        0.031621\n",
       "poutcome_failure        0.020714\n",
       "education_unknown       0.014355\n",
       "marital_divorced        0.005228\n",
       "job_self-employed      -0.004707\n",
       "job_technician         -0.011557\n",
       "job_entrepreneur       -0.034443\n",
       "default_cat            -0.040680\n",
       "job_pink-collar        -0.051717\n",
       "education_secondary    -0.051952\n",
       "education_primary      -0.063002\n",
       "marital_married        -0.092157\n",
       "job_blue-collar        -0.100840\n",
       "loan_cat               -0.110580\n",
       "campaign               -0.128081\n",
       "housing_cat            -0.203888\n",
       "poutcome_unknown       -0.224785"
      ]
     },
     "execution_count": 39,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Extract the deposte_cat column (the dependent variable)\n",
    "corr_deposite = pd.DataFrame(corr['deposit_cat'].drop('deposit_cat'))\n",
    "corr_deposite.sort_values(by = 'deposit_cat', ascending = False)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "metadata": {
    "id": "6CYDTafucsqs",
    "tags": []
   },
   "outputs": [],
   "source": [
    "# Train-Test split: 20% test data\n",
    "data_drop_deposite = bankcl.drop('deposit_cat', axis=1)\n",
    "label = bankcl.deposit_cat\n",
    "data_train, data_test, label_train, label_test = train_test_split(data_drop_deposite, label, test_size = 0.2, random_state = 50)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 41,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "oF8JFOEtcst1",
    "outputId": "5c981a65-c8a2-4594-8ba8-de294fbfc229",
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Training score:  0.7285250307985217\n",
      "Testing score:  0.7268248992386923\n"
     ]
    }
   ],
   "source": [
    "# Decision tree with depth = 2\n",
    "dt2 = tree.DecisionTreeClassifier(random_state=1, max_depth=2)\n",
    "dt2.fit(data_train, label_train)\n",
    "dt2_score_train = dt2.score(data_train, label_train)\n",
    "print(\"Training score: \",dt2_score_train)\n",
    "dt2_score_test = dt2.score(data_test, label_test)\n",
    "print(\"Testing score: \",dt2_score_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 42,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "OQSMl66xcswU",
    "outputId": "4743e97e-7b70-4a04-df45-607f4a07296b",
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Training score:  0.770411020271027\n",
      "Testing score:  0.7572772055530677\n"
     ]
    }
   ],
   "source": [
    "# Decision tree with depth = 3\n",
    "dt3 = tree.DecisionTreeClassifier(random_state=1, max_depth=3)\n",
    "dt3.fit(data_train, label_train)\n",
    "dt3_score_train = dt3.score(data_train, label_train)\n",
    "print(\"Training score: \",dt3_score_train)\n",
    "dt3_score_test = dt3.score(data_test, label_test)\n",
    "print(\"Testing score: \",dt3_score_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 43,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "46ecByDUcsyu",
    "outputId": "65e0b67b-7900-44dc-8cd6-50bfc124d3cb",
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Training score:  0.7885541494008288\n",
      "Testing score:  0.774294670846395\n"
     ]
    }
   ],
   "source": [
    "# Decision tree with depth = 4\n",
    "dt4 = tree.DecisionTreeClassifier(random_state=1, max_depth=4)\n",
    "dt4.fit(data_train, label_train)\n",
    "dt4_score_train = dt4.score(data_train, label_train)\n",
    "print(\"Training score: \",dt4_score_train)\n",
    "dt4_score_test = dt4.score(data_test, label_test)\n",
    "print(\"Testing score: \",dt4_score_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 44,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "B8jNh3nJcs1i",
    "outputId": "242aafb6-6a6e-42f9-d5d5-b6e9b2d95ace",
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Training score:  0.8080412140217269\n",
      "Testing score:  0.7796686072548141\n"
     ]
    }
   ],
   "source": [
    "# Decision tree with depth = 6\n",
    "dt6 = tree.DecisionTreeClassifier(random_state=1, max_depth=6)\n",
    "dt6.fit(data_train, label_train)\n",
    "dt6_score_train = dt6.score(data_train, label_train)\n",
    "print(\"Training score: \",dt6_score_train)\n",
    "dt6_score_test = dt6.score(data_test, label_test)\n",
    "print(\"Testing score: \",dt6_score_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 45,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "jUykI95hcs3o",
    "outputId": "f33fadba-06b6-424e-b568-1b71eeb49c82",
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Training score:  1.0\n",
      "Testing score:  0.7371249440214958\n"
     ]
    }
   ],
   "source": [
    "# Decision tree: To the full depth\n",
    "dt1 = tree.DecisionTreeClassifier()\n",
    "dt1.fit(data_train, label_train)\n",
    "dt1_score_train = dt1.score(data_train, label_train)\n",
    "print(\"Training score: \", dt1_score_train)\n",
    "dt1_score_test = dt1.score(data_test, label_test)\n",
    "print(\"Testing score: \", dt1_score_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 46,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "HqLZlVCKceUm",
    "outputId": "4d47b3ab-4d22-4088-82d9-dee92c8c2a1a",
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "depth      Training score       Testing score       \n",
      "-----      --------------       -------------       \n",
      "2        0.7285250307985217   0.7268248992386923\n",
      "3         0.770411020271027   0.7572772055530677\n",
      "4        0.7885541494008288    0.774294670846395\n",
      "6        0.8080412140217269   0.7796686072548141\n",
      "max                     1.0   0.7371249440214958\n"
     ]
    }
   ],
   "source": [
    "print('{:10} {:20} {:20}'.format('depth', 'Training score','Testing score'))\n",
    "print('{:10} {:20} {:20}'.format('-----', '--------------','-------------'))\n",
    "print('{:1} {:>25} {:>20}'.format(2, dt2_score_train, dt2_score_test))\n",
    "print('{:1} {:>25} {:>20}'.format(3, dt3_score_train, dt3_score_test))\n",
    "print('{:1} {:>25} {:>20}'.format(4, dt4_score_train, dt4_score_test))\n",
    "print('{:1} {:>25} {:>20}'.format(6, dt6_score_train, dt6_score_test))\n",
    "print('{:1} {:>23} {:>20}'.format(\"max\", dt1_score_train, dt1_score_test))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 47,
   "metadata": {
    "id": "hscECCDadA2D",
    "tags": []
   },
   "outputs": [],
   "source": [
    "# Let's generate the decision tree for depth = 2\n",
    "# Create a feature vector\n",
    "features = bankcl.columns.tolist()\n",
    "\n",
    "# Uncomment below to generate the digraph Tree.\n",
    "#tree.export_graphviz(dt2, out_file='tree_depth_2.dot', feature_names=features)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 48,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "fuGdYRMtdA8x",
    "outputId": "9acb4141-2d83-48e8-c97b-045d948cc3cc",
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([0, 1], dtype=int64)"
      ]
     },
     "execution_count": 48,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Two classes: 0 = not signed up,  1 = signed up\n",
    "dt2.classes_"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 49,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "UqzedZaYdA_W",
    "outputId": "e34d9d12-fa4b-4b5f-ab7d-ffa937979dff",
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "['age',\n",
       " 'balance',\n",
       " 'duration',\n",
       " 'campaign',\n",
       " 'previous',\n",
       " 'default_cat',\n",
       " 'housing_cat',\n",
       " 'loan_cat',\n",
       " 'recent_pdays',\n",
       " 'job_blue-collar',\n",
       " 'job_entrepreneur',\n",
       " 'job_other',\n",
       " 'job_pink-collar',\n",
       " 'job_self-employed',\n",
       " 'job_technician',\n",
       " 'job_white-collar',\n",
       " 'marital_divorced',\n",
       " 'marital_married',\n",
       " 'marital_single',\n",
       " 'education_primary',\n",
       " 'education_secondary',\n",
       " 'education_tertiary',\n",
       " 'education_unknown',\n",
       " 'poutcome_failure',\n",
       " 'poutcome_success',\n",
       " 'poutcome_unknown']"
      ]
     },
     "execution_count": 49,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Create a feature vector\n",
    "features = data_drop_deposite.columns.tolist()\n",
    "\n",
    "features"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 50,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "0EBnscaTdBBw",
    "outputId": "45fee556-a999-4b51-d6b3-24c5a1334661",
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "age................. 0.0\n",
      "balance............. 0.0\n",
      "duration............ 0.849306123902405\n",
      "campaign............ 0.0\n",
      "previous............ 0.0\n",
      "default_cat......... 0.0\n",
      "housing_cat......... 0.0\n",
      "loan_cat............ 0.0\n",
      "recent_pdays........ 0.0\n",
      "job_blue-collar..... 0.0\n",
      "job_entrepreneur.... 0.0\n",
      "job_other........... 0.0\n",
      "job_pink-collar..... 0.0\n",
      "job_self-employed... 0.0\n",
      "job_technician...... 0.0\n",
      "job_white-collar.... 0.0\n",
      "marital_divorced.... 0.0\n",
      "marital_married..... 0.0\n",
      "marital_single...... 0.0\n",
      "education_primary... 0.0\n",
      "education_secondary. 0.0\n",
      "education_tertiary.. 0.0\n",
      "education_unknown... 0.0\n",
      "poutcome_failure.... 0.0\n",
      "poutcome_success.... 0.15069387609759496\n",
      "poutcome_unknown.... 0.0\n"
     ]
    }
   ],
   "source": [
    "# Investigate most important features with depth =2\n",
    "\n",
    "dt2 = tree.DecisionTreeClassifier(random_state=1, max_depth=2)\n",
    "\n",
    "# Fit the decision tree classifier\n",
    "dt2.fit(data_train, label_train)\n",
    "\n",
    "fi = dt2.feature_importances_\n",
    "\n",
    "l = len(features)\n",
    "for i in range(0,len(features)):\n",
    "    print('{:.<20} {:3}'.format(features[i],fi[i]))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 51,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "XD03T_sodBEe",
    "outputId": "c1817871-2762-4ae5-c861-29fdf4bba146",
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Mean duration   :  371.99381831213043\n",
      "Maximun duration:  3881\n",
      "Minimum duration:  2\n"
     ]
    }
   ],
   "source": [
    "# According to feature importance results, most importtant feature is the \"Duration\"\n",
    "# Let's calculte statistics on Duration\n",
    "print(\"Mean duration   : \", data_drop_deposite.duration.mean())\n",
    "print(\"Maximun duration: \", data_drop_deposite.duration.max())\n",
    "print(\"Minimum duration: \", data_drop_deposite.duration.min())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 54,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "sBV-b4zXdT7-",
    "outputId": "e68389d6-cf76-48fd-90b0-8ce59e226763",
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "age                          46\n",
       "balance                    3354\n",
       "duration                    522\n",
       "campaign                      1\n",
       "previous                      1\n",
       "default_cat                   0\n",
       "housing_cat                   1\n",
       "loan_cat                      0\n",
       "recent_pdays           0.005747\n",
       "job_blue-collar           False\n",
       "job_entrepreneur          False\n",
       "job_other                  True\n",
       "job_pink-collar           False\n",
       "job_self-employed         False\n",
       "job_technician            False\n",
       "job_white-collar          False\n",
       "marital_divorced           True\n",
       "marital_married           False\n",
       "marital_single            False\n",
       "education_primary         False\n",
       "education_secondary        True\n",
       "education_tertiary        False\n",
       "education_unknown         False\n",
       "poutcome_failure          False\n",
       "poutcome_success           True\n",
       "poutcome_unknown          False\n",
       "Name: 985, dtype: object"
      ]
     },
     "execution_count": 54,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Get a row with poutcome_success = 1\n",
    "#bank_with_dummies[(bank_with_dummies.poutcome_success == 1)]\n",
    "data_drop_deposite.iloc[985]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 55,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "ausD6OdVdUBo",
    "outputId": "c042cf37-c012-447e-d407-6a2a56282349",
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "Accuracy score: \n",
      "0.7268248992386923\n",
      "\n",
      "Area Under Curve: \n",
      "0.7880265888143609\n"
     ]
    }
   ],
   "source": [
    "# Make predictions on the test set\n",
    "preds = dt2.predict(data_test)\n",
    "\n",
    "# Calculate accuracy\n",
    "print(\"\\nAccuracy score: \\n{}\".format(metrics.accuracy_score(label_test, preds)))\n",
    "\n",
    "# Make predictions on the test set using predict_proba\n",
    "probs = dt2.predict_proba(data_test)[:,1]\n",
    "\n",
    "# Calculate the AUC metric\n",
    "print(\"\\nArea Under Curve: \\n{}\".format(metrics.roc_auc_score(label_test, probs)))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "id": "JZYk29evdUER"
   },
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "id": "V5RDDei2dUGk"
   },
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "colab": {
   "provenance": []
  },
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
