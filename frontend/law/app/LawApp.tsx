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
import DocumentReader from '../pages/DocumentReader';
import MatterResearch from '../pages/MatterResearch';
import MatterCompare from '../pages/MatterCompare';
import MatterMeetings from '../pages/MatterMeetings';
import ResearchNote from '../pages/ResearchNote';
import Reviews from '../pages/Reviews';
import Evaluations from '../pages/Evaluations';
import Templates from '../pages/Templates';
import MatterDrafting from '../pages/MatterDrafting';
import DraftPage from '../pages/DraftPage';
import Playbooks from '../pages/Playbooks';
import DocumentPlaybook from '../pages/DocumentPlaybook';
import DocumentCitations from '../pages/DocumentCitations';
import DocumentDefinitions from '../pages/DocumentDefinitions';
import MatterObligations from '../pages/MatterObligations';
import MatterDiligence from '../pages/MatterDiligence';
import Checklists from '../pages/Checklists';
import RedlinePage from '../pages/RedlinePage';
import MatterMeeting from '../pages/MatterMeeting';
import MatterExports, { MatterExport } from '../pages/MatterExports';
import NotAvailable from '../pages/NotAvailable';
import DesignCatalogue from '../pages/DesignCatalogue';
import Enquiries from '../pages/Enquiries';
import Enquiry from '../pages/Enquiry';
import IntakeLink from '../pages/IntakeLink';
import MatterSummaries from '../pages/MatterSummaries';
import SummaryPage from '../pages/SummaryPage';
import TimeEntries from '../pages/TimeEntries';
import MatterPanel from '../pages/MatterPanel';
import PanelReviewPage from '../pages/PanelReviewPage';

function Root() {
  useSessionExpiry();
  return <Outlet />;
}

export const lawRoutes = [
  {
    element: <Root />,
    children: [
      { path: '/signin', element: <SignIn /> },
      { path: '/intake/:token', element: <IntakeLink /> },                // an enquirer's link: no session (S4.1, D-054)
      { path: '/', element: <Navigate to="/home" replace /> },
      ...(import.meta.env.DEV ? [{ path: '/__design', element: <DesignCatalogue /> }] : []),
      {
        element: <RequireSession><Shell /></RequireSession>,
        children: [
          { path: '/home', element: <Home /> },
          { path: '/matters', element: <Matters /> },
          { path: '/reviews', element: <Reviews /> },
          { path: '/templates', element: <Templates /> },
          { path: '/playbooks', element: <Playbooks /> },
          { path: '/checklists', element: <Checklists /> },
          { path: '/enquiries', element: <Enquiries /> },
          { path: '/enquiries/:intakeId', element: <Enquiry /> },
          { path: '/time', element: <TimeEntries /> },
          {
            path: '/m/:matterNo',
            element: <MatterLayout />,
            children: [
              { index: true, element: <Navigate to="overview" replace /> },
              { path: 'overview', element: <MatterOverview /> },
              { path: 'documents', element: <MatterDocuments /> },
              { path: 'documents/:docId/read', element: <DocumentReader /> },
              { path: 'documents/:docId/playbook', element: <DocumentPlaybook /> },
              { path: 'documents/:docId/citations', element: <DocumentCitations /> },
              { path: 'documents/:docId/definitions', element: <DocumentDefinitions /> },
              { path: 'redlines/:artifactId', element: <RedlinePage /> },
              { path: 'research', element: <MatterResearch /> },
              { path: 'compare', element: <MatterCompare /> },
              { path: 'meetings', element: <MatterMeetings /> },
              { path: 'meetings/:meetingId', element: <MatterMeeting /> },
              { path: 'notes/:artifactId', element: <ResearchNote /> },
              { path: 'drafting', element: <MatterDrafting /> },
              { path: 'obligations', element: <MatterObligations /> },
              { path: 'diligence', element: <MatterDiligence /> },
              { path: 'summaries', element: <MatterSummaries /> },
              { path: 'summaries/:artifactId', element: <SummaryPage /> },
              { path: 'panel', element: <MatterPanel /> },
              { path: 'panel/:reviewId', element: <PanelReviewPage /> },
              { path: 'drafting/:artifactId', element: <DraftPage /> },
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
          { path: '/admin/evaluations', element: <Evaluations /> },
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
