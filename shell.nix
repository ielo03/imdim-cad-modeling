{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  buildInputs = with pkgs; [
    # keep-sorted start
    gpp
    openscad
    just
    # keep-sorted end
  ];
}
