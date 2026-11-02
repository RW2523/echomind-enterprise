import { useEffect } from 'react';
import { useNavigate } from 'react-router';
import { useMatter } from '../app/matter';
import { useChatMatter } from '../counsel/context';

/** "Talk by voice" for a matter (its tab, the home, the command palette): voice mode in a new chat about it, as in
 *  ChatPage — the conversation assistant and the chat are one (owner's request 2026-11-02). */
export default function TalkByVoice() {
  const matter = useMatter();
  const { setMatterId } = useChatMatter();
  const navigate = useNavigate();
  useEffect(() => {
    setMatterId(matter.id);
    navigate('/chat', { replace: true, state: { voice: true } });
  }, [matter.id, setMatterId, navigate]);
  return null;
}
