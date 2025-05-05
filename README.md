# Imaging Physics Lab Toolbox
Toolbox for medical imaging data pre & post processing for the Meikle Imaging Physics Lab @ USYD

The repository currently includes two tools in the Jupyter Notebooks folder:
- Image derived input function automated extraction
    - Contains code for DICOM reading and preprocessing
    - Extracts input functions from either the aorta or left ventricle
    - Can be adapted for other segmentation tasks via MOOSE

- Vectorised Least Squares for the lp-ntPET model, see Normandin, M. D., Schiffer, W. K., & Morris, E. D. (2012). A linear model for estimation of neurotransmitter response profiles from dynamic PET data. NeuroImage, 59(3), 2689–2699. https://doi.org/10.1016/j.neuroimage.2011.07.002
    - Implemented to handle large datasets with GPU acceleration as an option
    - Use the basis function method with (weighted) least squares
    - A non-negative least squares has also been included. Vectorisation for it is not possible.
 
- A HYPR denoising implementation, see Christian, B. T., Vandehey, N. T., Floberg, J. M., & Mistretta, C. A. (2010). Dynamic PET Denoising with HYPR Processing. Journal of Nuclear Medicine, 51(7), 1147–1154. https://doi.org/10.2967/jnumed.109.073999
 
Most modules used can be installed via pip. For MOOSE, use ```pip install moosez```. See https://moosez.readthedocs.io/en/latest/installation.html#installing-via-pip.

I may or may not be adding more tools here depending on what I'll need during my PhD. If you wish to ask a question, report a bug or request a feature, please open an issue or just email me. If you have ideas for improvements or new features, feel free to contribute.
