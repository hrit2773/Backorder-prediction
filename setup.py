from setuptools import find_packages,setup
hyphen_e_dot='-e .'
def get_requirements(filepath):
    '''This function will return the list of requirements'''
    L=[]
    with open(filepath) as f:
        L=f.readlines()
        L=[req.replace("\n","") for req in L]
        if hyphen_e_dot in L:
            L.remove(hyphen_e_dot)
    return L    
        
        
setup(
    name='mlproject',
    version='0.0.1',
    author='Hritesh',
    author_email='shanty.hritesh@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)