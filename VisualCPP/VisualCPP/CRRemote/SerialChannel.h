#pragma once

#ifndef __SERIALCHANNEL_H__
#define __SERIALCHANNEL_H__

#ifndef CSERIALCHANNEL_EXT_CLASS
#define CSERIALCHANNEL_EXT_CLASS
#endif

#include "Channel.h"

#include "serialport.h"


#define __SHUTDOWN_TIMEOUT 5000


class CSERIALCHANNEL_EXT_CLASS CSerialChannel :
	public CChannel
{
public:
	CSerialChannel(void);
	~CSerialChannel(void);

    CString PortName() const;
	void SetPortName(const CString& value);

    BOOL IsOpen() const;
	int ReadTimeout();
	void SetReadTimeout(int timeout);


	void Open();
	void Close();

	void WriteLine(const CString& text);
	CString ReadExisting();

protected:
	CSerialPort m_serialPort;
	CString m_portName;
	CWinThread* m_readerThread;	
	CWinThread* m_writerThread;	
	//COMMTIMEOUTS m_CommTimeouts;
	//DCB m_dcb;
	CEvent m_writeEvent;
	CEvent m_shutdownEvent;
	BOOL m_isRunning;
	CStringArray m_szWriteBuffer;
	CStringArray m_szReadBuffer;

	CMutex m_syncObject;
private:
	CString TakeWriteBuffer();
	void OnDataReceived(const CString& buffer);
	void OnDataTransmitted(const CString& buffer);
	static UINT ReaderProc(LPVOID pParam);
	static UINT WriterProc(LPVOID pParam);
	void TransmitChar();
	void ReceiveChar();
	BOOL StartThreads();
	DWORD StopThreads(DWORD dwTimeout = __SHUTDOWN_TIMEOUT);
};

#endif //__SERIALCHANNEL_H__