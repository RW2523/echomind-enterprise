import React, { useEffect } from 'react';
import { createBrowserRouter, Navigate, Outlet, RouterProvider } from 'react-router';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import '@fontsource-variable/inter';
import '@fontsource-variable/source-serif-4';
import '../theme/law.css';
import { applyPreferences } from './preferences';
import { RequireSession, useSessionExpiry } from './session';
import { MatterLayout } from './matter';
import Shell from './Shell';
import SignIn from '../pages/SignIn';
import Home from '../pages/Home';
import Matters from '../pages/Matters';
import MatterOverview from '../pages/MatterOverview';
import MatterMembers from '../pages/MatterMembers';
import Security from '../pages/Security';
import AdminUsers from '../pages/AdminUsers';
import Walls from '../pages/Walls';
import EmergencyAccess from '../pages/EmergencyAccess';
import AuditTrail from '../pages/AuditTrail';
import Records from '../pages/Records';
import Connectors from '../pages/Connectors';
import MatterActivity from '../pages/MatterActivity';
import MatterDocuments from '../pages/MatterDocuments';
import MatterResearch from '../pages/MatterResearch';
import MatterMeetings from '../pages/MatterMeetings';
import MatterMeeting from '../pages/MatterMeeting';
import MatterExports, { MatterExport } from '../pages/MatterExports';
import NotAvailable from '../pages/NotAvailable';
import DesignCatalogue from '../pages/DesignCatalogue';

function Root() {
  useSessionExpiry();
  return <Outlet />;
}

export const lawRoutes = [
  {
    element: <Root />,
    children: [
      { path: '/signin', element: <SignIn /> },
      { path: '/', element: <Navigate to="/home" replace /> },
      ...(import.meta.env.DEV ? [{ path: '/__design', element: <DesignCatalogue /> }] : []),
      {
        element: <RequireSession><Shell /></RequireSession>,
        children: [
          { path: '/home', element: <Home /> },
          { path: '/matters', element: <Matters /> },
          {
            path: '/m/:matterNo',
            element: <MatterLayout />,
            children: [
              { index: true, element: <Navigate to="overview" replace /> },
              { path: 'overview', element: <MatterOverview /> },
              { path: 'documents', element: <MatterDocuments /> },
              { path: 'research', element: <MatterResearch /> },
              { path: 'meetings', element: <MatterMeetings /> },
              { path: 'meetings/:meetingId', element: <MatterMeeting /> },
              { path: 'exports', element: <MatterExports /> },
              { path: 'exports/:exportId', element: <MatterExport /> },
              { path: 'members', element: <MatterMembers /> },
              { path: 'activity', element: <MatterActivity /> },
              { path: '*', element: <NotAvailable /> },
            ],
          },
          { path: '/account/security', element: <Security /> },
          { path: '/admin/users', element: <AdminUsers /> },
          { path: '/admin/walls', element: <Walls /> },
          { path: '/admin/emergency-access', element: <EmergencyAccess /> },
          { path: '/admin/audit', element: <AuditTrail /> },
          { path: '/admin/records', element: <Records /> },
          { path: '/admin/connectors', element: <Connectors /> },
          { path: '*', element: <NotAvailable /> },
        ],
      },
    ],
  },
];

export function makeQueryClient() {
  return new QueryClient({
    defaultOptions: {
      queries: {
        staleTime: 15_000,
        refetchOnWindowFocus: true,
        retry: (count, err: any) => err?.status !== 401 && err?.status !== 403 && err?.status !== 404 && count < 2,
      },
    },
  });
}

/** The EchoMind Law workspace (S1.3). Loaded only when the backend reports the Law edition. */
export default function LawApp() {
  const [queryClient] = React.useState(makeQueryClient);
  const [router] = React.useState(() => createBrowserRouter(lawRoutes));
  useEffect(() => {
    document.documentElement.dataset.edition = 'law';
    document.title = 'EchoMind Law';
    applyPreferences();
    const mq = window.matchMedia?.('(prefers-color-scheme: dark)');
    const onChange = () => applyPreferences();
    mq?.addEventListener?.('change', onChange);
    return () => mq?.removeEventListener?.('change', onChange);
  }, []);
  return (
    <QueryClientProvider client={queryClient}>
      <RouterProvider router={router} />
    </QueryClientProvider>
  );
}
