import numpy as np
import matplotlib.pyplot as plt
from colorama import Fore, Style
import os

def operation_practice():
    vector = np.array([1,2,3])
    word_vector = np.array([0.2,-0.5,0.8,-0.1,0.3])
    print(f"Vector:{vector},shape:{vector.shape[0]},Type:{type(vector)}")
    
    v1 = np.array([2, 3, 1])
    v2 = np.array([1, 2, 3])
    #Addition 
    print(v1+v2)
    #Dot product 这种适合一维的情况，如果是二维的，那用matmul或者@会更好
    print(np.dot(v1,v2))

    vector = np.array([3, 4])
    magnitude = np.linalg.norm(vector)
    print(f"||{vector}|| = {magnitude}")

    normalized = vector/magnitude
    print(f"{np.linalg.norm(normalized):.6f}")

    v1 = np.array([1, 1, 0])
    v2 = np.array([1, 0, 0])
    v3 = np.array([0, 1, 1])

    def cosine_similarity(a,b):
        dot_product = np.dot(a,b)
        magnitude_a = np.linalg.norm(a)
        magnitude_b = np.linalg.norm(b)
        return dot_product/(magnitude_a*magnitude_b)
    
    sim_12 = cosine_similarity(v1,v2)
    sim_13 = cosine_similarity(v1,v3)
    sim_32 = cosine_similarity(v3,v2)


    print(f"Cosine similarity (v1, v2): {sim_12:.4f}")
    print(f"Cosine similarity (v1, v3): {sim_13:.4f}")
    print(f"Cosine similarity (v2, v3): {sim_32:.4f}\n")

def visualize_2d_vectors():
    os.makedirs("outputs",exist_ok = True)

    vectors = {
        'v1': np.array([3, 2]),
        'v2': np.array([1, 3]),
        'v3': np.array([-2, 1]),
        'v4': np.array([2, -2])
    }

    plt.figure(figsize=(10,10))
    plt.axhline(y=0,color = 'k',linewidth =0.5)
    plt.axvline(x=0,color = 'k',lindwidth =0.5)
    plt.grid(True,alpha=0.3)
    
    colors = ['red','blue','green','purple']
    for (name,vec), color in zip(vectors.items(),colors):
        plt.arrow(0,0,vec[0],vec[1],head_width=0.3,head_length=0.3,
                  fc=color,ec=color,linewidth=2,label=name)
        plt.text(vec[0]+0.3,vec[1]+0.3,name,fontsize=12,fontweight='bold')

    plt.xlim(-4,4)
    plt.ylim(-4,4)
    plt.xlabel('Dimension 1',fontsize =12)
    plt.ylabel('Dimension 2',fontsize =12)
    plt.title('2D Vector Visualization', fontsize=14, fontweight='bold')
    plt.legend()
    plt.savefig('outputs/module1_vectors_2d.png', dpi=150, bbox_inches='tight')
    print("✓ Saved visualization to outputs/module1_vectors_2d.png\n")
    plt.close()

if __name__ == "__main__" :
    operation_practice()