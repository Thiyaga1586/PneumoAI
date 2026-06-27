const IST_TIME_ZONE = 'Asia/Kolkata';
const MAINTENANCE_START_MINUTES = 21 * 60; // 9:00 PM
const MAINTENANCE_END_MINUTES = 7 * 60 + 15; // 7:15 AM

function getIstMinutes(now: Date = new Date()): number {
  const parts = new Intl.DateTimeFormat('en-GB', {
    timeZone: IST_TIME_ZONE,
    hour: '2-digit',
    minute: '2-digit',
    hour12: false,
  }).formatToParts(now);

  const hour = Number(parts.find((p) => p.type === 'hour')?.value ?? '0');
  const minute = Number(parts.find((p) => p.type === 'minute')?.value ?? '0');

  return hour * 60 + minute;
}

export function isWithinMaintenanceWindow(now: Date = new Date()): boolean {
  const minutes = getIstMinutes(now);
  return minutes >= MAINTENANCE_START_MINUTES || minutes < MAINTENANCE_END_MINUTES;
}

export function maintenanceWindowLabel(): string {
  return '9:00 PM–7:15 AM IST';
}