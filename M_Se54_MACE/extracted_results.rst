
Hg@Se54
-------

full relax
~~~~~~~~~~
(mace_env) miroi@MIRO:~/work/projects/papers/MolPhys2026-codes/M_Se54_MACE/.python visualize_vibrations.py --structure Hg_Se_opt.traj --phonon-dir adsorption_se54_dip_Hg_slab_phonons

  mode  23:     31.010 cm^-1, Hg projection = 0.3520, Hg perpendicular fraction = 0.959, perpendicular weight = 1.690e-01
  ase gui Hg_vibrations/Hg_mode_031.010_cm-1.traj
 
frozen slab
~~~~~~~~~~~
Hg-dominated frequencies (cm^-1): [11.974 14.587 29.871]
Top 3 modes by Hg perpendicular (z) projection:
  mode   2:     29.871 cm^-1; total adsorbate projection=1.0000; z projection=0.9775; z fraction of adsorbate motion=0.9775
  mode   0:     11.974 cm^-1; total adsorbate projection=1.0000; z projection=0.0174; z fraction of adsorbate motion=0.0174
  mode   1:     14.587 cm^-1; total adsorbate projection=1.0000; z projection=0.0051; z fraction of adsorbate motion=0.0051

(mace_env) miroi@MIRO:~/work/projects/papers/MolPhys2026-codes/M_Se54_MACE/.python visualize_vibrations.py --structure Hg_Se_opt.traj --phonon-dir adsorption_se54_dip_Hg_slab_phonons
  mode   0:     11.974 cm^-1, Hg projection = 1.0000, Hg perpendicular fraction = 0.017, perpendicular weight = 1.738e-02
  mode   1:     14.587 cm^-1, Hg projection = 1.0000, Hg perpendicular fraction = 0.005, perpendicular weight = 5.126e-03
  mode   2:     29.871 cm^-1, Hg projection = 1.0000, Hg perpendicular fraction = 0.977, perpendicular weight = 9.775e-01 <----------------

  ase gui Hg_vibrations/Hg_mode_029.871_cm-1.traj


Pb@Se54
-------

full relax
~~~~~~~~~~
Top 10 modes by Pb perpendicular (z) projection:
  mode 113:    148.875 cm^-1; total adsorbate projection=0.0623; z projection=0.0345; z fraction of adsorbate motion=0.5539
  mode   5:     11.017 cm^-1; total adsorbate projection=0.0238; z projection=0.0226; z fraction of adsorbate motion=0.9511 <----------
  mode  71:     56.291 cm^-1; total adsorbate projection=0.0202; z projection=0.0158; z fraction of adsorbate motion=0.7796
  mode  64:     51.912 cm^-1; total adsorbate projection=0.0200; z projection=0.0157; z fraction of adsorbate motion=0.7852
  mode 111:    135.759 cm^-1; total adsorbate projection=0.0265; z projection=0.0152; z fraction of adsorbate motion=0.5729
  mode  45:     42.875 cm^-1; total adsorbate projection=0.0331; z projection=0.0151; z fraction of adsorbate motion=0.4556
  mode  43:     39.334 cm^-1; total adsorbate projection=0.0395; z projection=0.0140; z fraction of adsorbate motion=0.3536
  mode 109:    124.438 cm^-1; total adsorbate projection=0.0661; z projection=0.0137; z fraction of adsorbate motion=0.2075
  mode  29:     30.130 cm^-1; total adsorbate projection=0.0143; z projection=0.0127; z fraction of adsorbate motion=0.8862
  mode   4:      9.915 cm^-1; total adsorbate projection=0.0275; z projection=0.0125; z fraction of adsorbate motion=0.4544

Selected modes:
  mode 113:    148.875 cm^-1, Pb projection = 0.1485, Pb perpendicular fraction = 0.554, perpendicular weight = 3.451e-02
  mode   5:     11.017 cm^-1, Pb projection = 0.0601, Pb perpendicular fraction = 0.951, perpendicular weight = 2.261e-02 <-----------
  mode  71:     56.291 cm^-1, Pb projection = 0.0514, Pb perpendicular fraction = 0.780, perpendicular weight = 1.578e-02

frozen slab
~~~~~~~~~~~
Selected modes:
  mode   1:     72.810 cm^-1, Pb projection = 1.0000, Pb perpendicular fraction = 0.559, perpendicular weight = 5.586e-01  <------
  mode   0:     59.463 cm^-1, Pb projection = 1.0000, Pb perpendicular fraction = 0.054, perpendicular weight = 5.386e-02
  mode   2:     81.120 cm^-1, Pb projection = 1.0000, Pb perpendicular fraction = 0.388, perpendicular weight = 3.875e-01

Pb@slab vibrational thermodynamics (ASE HarmonicThermo) - MACE
   Temp(K)           ZPE(eV)             F(eV)             U(eV)           S(eV/K)
    298.15      0.0132287066     -0.0826612970      0.0778448028  5.3834009634e-04

vs QE+D4, frozen slab
https://github.com/dipayanster/backup/blob/master/d4qe_ads3/5.1_New_Vibration/PBE/Pb/log_slurm_job.1821502.n05p030.std_out_err

 Total Energy (QE + D4):  -180038.414163 eV

Temperature: 298.15 K
  ZPE =   0.013196 eV  QE+D4: 0.0132287 eV
  F   = -180038.497785 eV - (-180038.414163) eV = -0.083622 eV QE+D4:  -0.08266 eV
  U   = -180038.336307 eV - (-180038.414163) eV = 0.077856 eV  QE+D4:   0.0778448 eV
  S   = 5.415996e-04 eV/K  QE+D4:  5.3834009634e-04
