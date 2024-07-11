.. Reformulator documentation master file, created by
   sphinx-quickstart on Thu Jul  6 07:29:34 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

General Documentation
======================

The Main Screen
----------------

The main screen of the program provides a simplified version of HSPiP's evaporation simulator, with a few additional features like temperature ramping.
To use the simulator, you'll need a database file with information about all of your solvents. Additional information about the file format can be found here.
You'll also need a file describing your formulation, with the first column (header "Name") being a list of solvent names and the second column 
(header "Weight Fraction" or "Volume Fraction") being a list of concentrations for each solvent.
Parameters for the target formulation & cure process are entered next, including the Hansen Solubility Parameters (dD, dP, and dH, and the radius R). 
If you don't have these for your target resin you can either try to approximate them with a similar resin or (preferably) ask Analytical to determine them.
Once everything is entered, you can click on "Run Simulation" to see results and export data using the "Export to Excel" button. 
Data on the graphs can be further analyzed using the pan & zoom tools provided in the toolbar at the bottom of the graph window.
If you click "Save Config" and save a .json file of your configuration, you can then load all of your inputs in using "Load Config" next time you run the program.

The Comparison Screen
-----------------------

To launch the comparison screen, click on "Compare Outputs" on the main screen.
Load in output files exported from one of the other screens using the "Select Output Files" button.
Select which aspects to compare using the checkboxes and click "Compare" to bring up the output.

The Reformulation Screen
--------------------------

To launch the reformulation screen, click on "Reformulate" on the main screen.
Load in a control blend (same format as formulation on main screen) and a minimum composition file (same format, specifying how much of the solvent package can't be changed).
For example, a minimum composition file saying .5 Weight Fraction t-Butyl Acetate would make the program produce output blends with at least 50% TBAC by weight.
Also, an important note: if your minimum composition file is based on Weight Fraction, the other components will be replaced by Weight Fraction (the same goes for Volume Fraction).
Optionally, include a whitelist (file containing complete list of allowed solvents) or blacklist (file containing list of disallowed solvents).
Add a solvent information database (same format as and synced with the main window.
You can then specify a maximum amount of non-exempt solvents by weight % or volume % (must match minimum composition file) or alternatively in lb/gal or g/L. 
Note: To maintain linearity, VOC constraints are in lb non-exempt/gal solids instead of the traditional lb non-exempt/gal (non-exempt + solids). 
This means results will be slightly under the speficied constraint. If you have issues with this reach out to whoever is maintaining the program and they can probably implement some nonlinear constraints.
Next, select the minimum and maximum number (both inclusive) of exempt & non-exempt solvents to include in blends.
Enter the target and temperature parameters as before, as well as the formulation density (if VOC limits were set in lb/gal or g/L).
Finally, click "Find Solvent Blends" to determine the best combinations of solvents for your target parameters.



Background and Equation Info (for non-programmers)
==================================================

Evaporation Equations 
---------------------

The primary differential equation used to simulate evaporation was reverse-engineered from HSPiP output:

.. math::

	\frac{dy_{i}}{dt} = -\frac{y_{i}}{\sum{y_{i}}} * RER(T(t)) * R_{nBA}

| Where :math:`y_{i}` is the total amount of component i remaining (as a fraction of the original amount of total solvent).
| So :math:`\sum{y_{i}}` is the total profile and :math:`\frac{y_{i}}{\sum{y_{i}}}` is the partial profile of component i.
| 
| Often (such as when suggesting solvent blends), it is too computationally expensive to use this differential equation, as the results have to be checked and optimized many times.
| So instead we can use an iterative shortcut. Notice that 

.. math::

	\frac{dy_{i}}{dt} = -\frac{RER(T(t)) * R_{nBA}}{\sum{y_{i}}} * y_{i}\\
	\frac{1}{y_{i}} dy_{i} = -\frac{RER(T(t)) * R_{nBA}}{\sum{y_{i}}} dt\\
	ln(y_{i}) = - RER(T(t)) * R_{nBA} * \int \frac{1}{\sum{y_{i}}} dt\\
	y_{i} = exp(- RER(T(t)) * R_{nBA} * \int_{0}^{t} \frac{1}{\sum{y_{i}}} dt

| So if we have an esimate of the total profile :math:`\sum{y_{i}}`, we can then estimate each partial profile as

.. math::

	y_{i}(t) = y_{i0} * e^{- RER(T(t)) * R_{nBA} * \int_{0}^{t} \frac{1}{\sum{y_{i}}}}

| We can then use these partial profiles to create a new estimate of the total profile :math:`\sum{y_{i}}`. Iterating this process twice seems to produce a reasonable estimate for the total evaporation profile.
| The only remaining question is how to get an initial estimate of the total profile, which I solve by assuming it's constant at 1 (a bad assumption but gives decent first partial profiles).
	

Cost Functions
--------------

| In order to compare potential new solvent blends to an existing blend, I defined a cost function based on three paramaters: Evaporation rates, Hansen Solubility, and VOC content.
| The current equation used for each is as follows:

* Evaporation Cost: Sum-squared error between min(test, 0.05) and min(control, 0.05) plus the square the net area (integral between) those two curves. Prioritizes late-stage evaporation.
* Hansen Solubility Cost: :math:`\sum{\frac{1}{1.01-RED(t)} * max(total\_profile(t), 0.01)}/\sum{total\_profile(t)}`, where RED is the RED of the test blend. This gives greater weight to initial solubility but significant weight to final solubility as well. Typical cost should be around 3-5 for a good solvent.
* VOC Cost: Designed to add some cushion to our VOC limit, this is defined as :math:`100 * (max(voc\_content/voc\_max, 0.9) - 0.9)^2`. Cost is always between 0 (less than 90% VOC maximum) and 1 (exactly at VOC maximum). Since VOC maximum is imposed as a constraint, 

| These cost functions should be good rough estimates of how well a solvent blend will perform, but have not been extensively tested. For this reason, I have made it easy to modify them.
| The functions evap_cost, hansen_cost, and voc_cost are all documented in section.

Optimization
------------

The total cost (sum of the 3 cost functions described above) is minimized in23 steps.

#. Exempt solvents only (iterative evaporation model)
#. All solvents (iterative evaporation model)

| In the first step, exempt solvents (and mixtures if the user specified more than 1 exempt being allowed) that absolutely will not work are eliminated from consideration.
| In the second step, all blends are screened with the approximate evaporation model to quickly eliminate full blends that won't work well (doing this with the full model could take a very long time as there could be thousands of potential blends).


API Documentation (for programmers)
===================================

Simulator
---------

.. automodule:: simulator
	:members:
	:undoc-members:

UI
--

.. automodule:: ui
	:members:
	:undoc-members:


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
