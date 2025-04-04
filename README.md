# Imaging Physics Lab Toolbox
Toolbox for medical imaging data pre & post processing

Currently contains two tools in the Jupyter Notebook folder:
- Image derived input function extraction
    - Contains code for DICOM reading and preprocessing
    - From either aorta or left ventricle
    - Can also be adapted for other segmentation tasks
- Vectorised Least Squares for the lp-ntPET model, see Normandin, M. D., Schiffer, W. K., & Morris, E. D. (2012). A linear model for estimation of neurotransmitter response profiles from dynamic PET data. NeuroImage, 59(3), 2689–2699. https://doi.org/10.1016/j.neuroimage.2011.07.002
    - Implemented to handle large datasets with GPU accleration as an option
    - Use the basis function method with (weighted) least squares
 
Most modules used can be installed via pip. For MOOSE, use pip install moosez. See https://moosez.readthedocs.io/en/latest/installation.html#installing-via-pip.

I may or may not be adding more tools here depending on what I'll need during my PhD. But also feel free to open a feature request / Q&A on Issues or just email me.
