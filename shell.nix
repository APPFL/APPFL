{ pkgs ? import (fetchTarball "https://github.com/NixOS/nixpkgs/tarball/nixos-23.11") {} }:

pkgs.mkShellNoCC {
  packages = with pkgs; [
    (python3.withPackages (ps: [
      ps.pip          # Include pip
      ps.tkinter
      ps.plotly
      ps.flask
      ps.numpy
      ps.matplotlib
      ps.pandas
      ps.mpi4py
      ps.torch
      ps.omegaconf
    ]))
    mpich            # MPI implementation
    gcc              # Includes libstdc++
    curl
    jq
  ];

  shellHook = ''
    export LD_LIBRARY_PATH=${pkgs.gcc}/lib:$LD_LIBRARY_PATH
  '';
}

