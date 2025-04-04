#!/bin/bash

for input in {0..10}
do
  echo $input
  python investigate_GCD_formation.py Halo1445_fiducial_hires $input
done

for input in {0..10}
do
  echo $input
  python investigate_GCD_formation.py Halo1459_fiducial_hires $input
done

for input in {0..100}
do
  echo $input
  python investigate_GCD_formation.py Halo600_fiducial_hires $input
done

for input in {0..100}
do
  echo $input
  python investigate_GCD_formation.py Halo605_fiducial_hires $input
done

for input in {0..100}
do
  echo $input
  python investigate_GCD_formation.py Halo624_fiducial_hires $input
done

for input in {0..200}
do
  echo $input
  python investigate_GCD_formation.py Halo383_fiducial_early $input
done

for input in {0..300}
do
  echo $input
  python investigate_GCD_formation.py Halo383_Massive $input
done
