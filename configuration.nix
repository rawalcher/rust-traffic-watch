{ config, pkgs, lib, ... }:

let
  user = "nixos";
  password = "nixos";
  SSID = "Das Internetz";
  wifiPassword = lib.strings.removeSuffix "\n" (builtins.readFile /etc/nixos/secrets/wifi-password);
  interface = "wlan0";
  hostname = "nixos-pi1";
in {

  boot = {
    kernelPackages = pkgs.linuxKernel.packages.linux_rpi4;
    initrd.availableKernelModules = [ "xhci_pci" "usbhid" "usb_storage" ];
    loader = {
      grub.enable = false;
      generic-extlinux-compatible.enable = true;
    };
  };

  fileSystems = {
    "/" = {
      device = "/dev/disk/by-label/NIXOS_SD";
      fsType = "ext4";
      options = [ "noatime" ];
    };
  };

  swapDevices = [
   { device = "/swapfile"; size = 4096; }
  ];

  networking = {
    hostName = hostname;
    wireless = {
      enable = true;
      networks."${SSID}".psk = wifiPassword;
      interfaces = [ interface ];
    };
  };

  environment.systemPackages = with pkgs; [ vim ];

  services.openssh.enable = true;

  users = {
    mutableUsers = false;
    users."${user}" = {
      isNormalUser = true;
      password = password;
      extraGroups = [ "wheel" ];
    };
  };

  hardware.enableRedistributableFirmware = true;
  system.stateVersion = "23.11";
}