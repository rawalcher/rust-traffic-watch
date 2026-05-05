{
  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  outputs = { nixpkgs, ... }: let
    pkgs = import nixpkgs {
      system = "x86_64-linux";
      config.permittedInsecurePackages = [ "electron-38.8.4" ];
    };
    rEnv = pkgs.rstudioWrapper.override {
      packages = with pkgs.rPackages; [ tidyverse data_table knitr rmarkdown Cairo gt webshot2];
    };
  in {
    devShells.x86_64-linux.default = pkgs.mkShell {
      packages = [ rEnv pkgs.quarto ];
    };
  };
}

