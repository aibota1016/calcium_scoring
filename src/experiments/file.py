from tqdm import tqdm

def f():
    for i in tqdm(range(10)):
        print(i)
    
    
    
if __name__ == '__main__':
    f()