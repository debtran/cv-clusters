# cv-clusters

This algorithm is used to find clusters using resolved or semi-resolved stellar photometry depending on the quality cuts applied to your photometric catalog. I use a modified version of the Canny Edge Detection algorithm on stellar density maps to obtain the edges of star clusters based on their stellar densities. The paper associated with this work is submitted to ApJ and contains further details on how the algorithm operates. It will be linked here when it is published.

In cluster-example.ipynb, the quality cuts made are for sources with a high signal to noise ratio to include blends of multiple stars at the dense star cluster centers. The data used in the example notebook are synthetic clusters inserted into a field of varying stellar densities in NGC 6946.

Please see https://heasarc.gsfc.nasa.gov/FTP/software/fitsio/c/docs/fpackguide.pdf for unpacking the .fz compressed fits file.
