import React from 'react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';

vi.mock('../services/backend', () => ({
  login: vi.fn(),
  mfaVerify: vi.fn(),
  mfaEnroll: vi.fn(),
  mfaConfirm: vi.fn(),
}));
import * as api from '../services/backend';
import Login from './Login';

const user = { id: 'usr_1', username: 'a.rahman', role: 'user' };

function signIn() {
  fireEvent.change(screen.getByLabelText('Username'), { target: { value: 'a.rahman' } });
  fireEvent.change(screen.getByLabelText('Password'), { target: { value: 'correct horse battery staple' } });
  fireEvent.click(screen.getByRole('button', { name: 'Sign in' }));
}

describe('Login', () => {
  afterEach(() => { cleanup(); vi.clearAllMocks(); });

  it('signs straight in when no second step is needed (legacy editions)', async () => {
    (api.login as any).mockResolvedValue({ status: 'ok', user });
    const onSuccess = vi.fn();
    render(<Login onSuccess={onSuccess} />);
    signIn();
    await waitFor(() => expect(onSuccess).toHaveBeenCalledWith(user));
  });

  it('asks for an authenticator code when two-step sign-in is on', async () => {
    (api.login as any).mockResolvedValue({ status: 'mfa_required' });
    (api.mfaVerify as any).mockResolvedValue(user);
    const onSuccess = vi.fn();
    render(<Login onSuccess={onSuccess} />);
    signIn();
    const code = await screen.findByLabelText('Authenticator code');
    expect(screen.queryByLabelText('Password')).toBeNull();
    fireEvent.change(code, { target: { value: '123456' } });
    fireEvent.click(screen.getByRole('button', { name: 'Verify' }));
    await waitFor(() => expect(onSuccess).toHaveBeenCalledWith(user));
    expect(api.mfaVerify).toHaveBeenCalledWith('123456');
  });

  it('walks a first sign-in through authenticator set-up and shows recovery codes once', async () => {
    (api.login as any).mockResolvedValue({ status: 'mfa_enrollment_required' });
    (api.mfaEnroll as any).mockResolvedValue({ secret: 'ABCDEFGHIJKLMNOP', otpauth_uri: 'otpauth://totp/x' });
    (api.mfaConfirm as any).mockResolvedValue({ user, recovery_codes: ['aaaaa-bbbbb', 'ccccc-ddddd'] });
    const onSuccess = vi.fn();
    render(<Login onSuccess={onSuccess} />);
    signIn();
    expect((await screen.findByTestId('mfa-secret')).textContent).toBe('ABCD EFGH IJKL MNOP');
    fireEvent.change(screen.getByLabelText('Authenticator code'), { target: { value: '654321' } });
    fireEvent.click(screen.getByRole('button', { name: 'Turn on two-step sign-in' }));
    const codes = await screen.findByTestId('recovery-codes');
    expect(codes.textContent).toContain('aaaaa-bbbbb');
    expect(onSuccess).not.toHaveBeenCalled();
    fireEvent.click(screen.getByRole('button', { name: 'I have saved my recovery codes' }));
    expect(onSuccess).toHaveBeenCalledWith(user);
  });

  it('shows the server message on failure without revealing which part was wrong', async () => {
    (api.login as any).mockRejectedValue(new Error('Sign-in failed. Check your username and password, or try again later.'));
    render(<Login onSuccess={vi.fn()} />);
    signIn();
    expect((await screen.findByRole('alert')).textContent).toMatch(/Sign-in failed/);
  });
});
