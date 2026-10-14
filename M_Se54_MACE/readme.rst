
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
