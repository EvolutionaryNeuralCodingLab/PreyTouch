# Configure Touch Screen for using the Application

## Screen Variables
Add the following variables to the .env file of the system
```bash
# The name to look for in the output of the command "xinput"
TOUCH_SCREEN_NAME=Elo
# screen resolution with "," between the dimenstions in pixels
SCREEN_RESOLUTION="1920,1080"
# Screen display id
APP_SCREEN=:0
# test display id, for running test experiments
TEST_SCREEN=:1
# if you need the screen to be inverted set this to 1
IS_SCREEN_INVERTED=0
```
## Security
### Auto-login
User for display must have autologin. 
For example if using gdm3 and the user name is ep-arena, edit the following file:
```console
vim /etc/gdm3/custom.conf
```
And add the following to allow auto-login for ep-arena:
```console
[daemon]
AutomaticLoginEnable=true
AutomaticLogin=ep-arena
WaylandEnable=false

[security]
DisallowTCP=false
AllowRemoteRoot=true
```
### Fix for Ubuntu color profile
If you get the following prompt "ubuntu autenctication is required to create a color profile", then do the following steps.

Create the following file:
```console
sudo vim /etc/polkit-1/localauthority/50-local.d/45-allow-colord.pkla
```
And paste the following in it:
```console
[Allow Colord all Users]
Identity=unix-user:*
Action=org.freedesktop.color-manager.create-device;org.freedesktop.color-manager.create-profile;org.freedesktop.color-manager.delete-device;org.freedesktop.color-manager.delete-profile;org.freedesktop.color-manager.modify-device;org.freedesktop.color-manager.modify-profile
ResultAny=no
ResultInactive=no
ResultActive=yes
```