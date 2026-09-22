#!/usr/bin/env bash
# Install a nightly systemd timer for scripts/backup_data.sh.
#
# ISO 9001:2015 8.5.4 — closes gap G-22 (backup existed but was manual and on-host).
#
#   sudo ./scripts/install_backup_timer.sh
#   sudo BACKUP_REMOTE=user@nas:/srv/echomind ./scripts/install_backup_timer.sh
#   sudo BACKUP_AT=03:30 ./scripts/install_backup_timer.sh
#
# Uninstall:  sudo systemctl disable --now echomind-backup.timer
set -euo pipefail

[ "$(id -u)" -eq 0 ] || { echo "ERROR: run with sudo" >&2; exit 1; }
command -v systemctl >/dev/null || { echo "ERROR: systemd not available — use cron instead (see foot of this file)" >&2; exit 1; }

REPO="$(cd "$(dirname "$0")/.." && pwd)"
RUN_AS="${SUDO_USER:-root}"
AT="${BACKUP_AT:-02:30}"
KEEP="${BACKUP_KEEP:-7}"
REMOTE="${BACKUP_REMOTE:-}"

[ -x "$REPO/scripts/backup_data.sh" ] || { echo "ERROR: $REPO/scripts/backup_data.sh not executable" >&2; exit 1; }

cat > /etc/systemd/system/echomind-backup.service <<UNIT
[Unit]
Description=EchoMind customer-data backup
Documentation=file://$REPO/docs/qms/procedures/SOP-11_Customer_Property_and_Data_Handling.md
After=docker.service
Requires=docker.service

[Service]
Type=oneshot
User=$RUN_AS
WorkingDirectory=$REPO
Environment=BACKUP_KEEP=$KEEP
$( [ -n "$REMOTE" ] && echo "Environment=BACKUP_REMOTE=$REMOTE" )
ExecStart=$REPO/scripts/backup_data.sh
# A failed backup must be visible, not silent.
StandardOutput=journal
StandardError=journal
UNIT

cat > /etc/systemd/system/echomind-backup.timer <<UNIT
[Unit]
Description=Nightly EchoMind customer-data backup

[Timer]
OnCalendar=*-*-* $AT:00
Persistent=true

[Install]
WantedBy=timers.target
UNIT

systemctl daemon-reload
systemctl enable --now echomind-backup.timer

echo "Installed."
echo "  runs as    : $RUN_AS"
echo "  schedule   : daily at $AT (Persistent=true — a missed run fires at next boot)"
echo "  keep       : $KEEP generations"
echo "  off-host   : ${REMOTE:-NOT SET — archives stay on this host, which is only half a backup}"
echo
systemctl list-timers echomind-backup.timer --no-pager | sed 's/^/  /'
echo
echo "Check it:      systemctl status echomind-backup.service"
echo "Run it now:    sudo systemctl start echomind-backup.service"
echo "Read the log:  journalctl -u echomind-backup.service -n 50"
echo
echo "Record the next verified restore against objective QO-4 (SOP-11 §8)."

# Cron equivalent, if systemd is unavailable:
#   30 2 * * *  cd /path/to/repo && BACKUP_REMOTE=user@nas:/srv/echomind ./scripts/backup_data.sh >> /var/log/echomind-backup.log 2>&1
