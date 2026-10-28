/**
 * Where answers come from, in words the configuration supports (commercial-readiness review P1, 2026-10-28). The server
 * reports `model_location` from the model's configured address; the interface never claims more than that, and never
 * that "nothing leaves": people can download and export.
 */
import type { Me } from '../api/types';

export interface Where { label: string; chip: string; title: string; text: string; external: boolean }

const RECORDED = 'Every question, source opened and decision is recorded in the audit trail.';
const SCOPE = 'only from the documents of the matter you chose, and only those you may read.';

export function whereAnswersComeFrom(me: Me): Where {
  switch (me.features?.model_location) {
    case 'this_server':
      return { label: 'Private', chip: 'On this server', title: 'Answered on this server', external: false,
        text: `Answers are written by the model running on this server, ${SCOPE} They are not sent to an outside AI service. ${RECORDED}` };
    case 'firm_network':
      return { label: 'Private', chip: 'On your network', title: "Answered on your firm's network", external: false,
        text: `Answers are written by a model on your firm's own network, ${SCOPE} They are not sent to an outside AI service. ${RECORDED}` };
    case 'external':
      return { label: 'External model', chip: 'External model', title: 'Answered by an external model service', external: true,
        text: `Answers are written by a model service outside your firm's network, chosen by your administrator, ${SCOPE} ` +
          `The passages needed for each answer are sent to that service. ${RECORDED}` };
    default:
      return { label: 'Sources', chip: '', title: 'Where answers come from', external: false, text: `Answers are written ${SCOPE} ${RECORDED}` };
  }
}
