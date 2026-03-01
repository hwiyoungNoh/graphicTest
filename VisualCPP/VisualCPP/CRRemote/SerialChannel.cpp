#include "StdAfx.h"
#include "SerialChannel.h"

CSerialChannel::CSerialChannel(void):
m_writeEvent(FALSE, FALSE),
m_shutdownEvent(FALSE, TRUE)
{
	m_portName = "";

	m_writerThread = NULL;
	m_readerThread = NULL;

}

CSerialChannel::~CSerialChannel(void)
{
	if(IsOpen())
	{
		Close();
	}
	
}

BOOL CSerialChannel::StartThreads()
{
	m_readerThread = AfxBeginThread(CSerialChannel::ReaderProc, (LPVOID)this, THREAD_PRIORITY_NORMAL, 0, CREATE_SUSPENDED, NULL);  
	if(!m_readerThread)
		return FALSE;
	m_writerThread = AfxBeginThread(CSerialChannel::WriterProc, (LPVOID)this, THREAD_PRIORITY_NORMAL, 0, CREATE_SUSPENDED, NULL);  
	if(!m_writerThread)
		return FALSE;

	m_readerThread->ResumeThread();
	m_writerThread->ResumeThread();


	return TRUE;;
}

DWORD CSerialChannel::StopThreads(DWORD dwTimeout)
{
    HANDLE hThreads[2];
    DWORD  dwRes;

	if(!m_readerThread && !m_readerThread)
		return 0;

    hThreads[0] = (HANDLE)m_readerThread;
    hThreads[1] = (HANDLE)m_writerThread;

    //
    // set thread exit event here
    //
    m_shutdownEvent.SetEvent();

    dwRes = WaitForMultipleObjects(2, hThreads, TRUE, dwTimeout);
    switch(dwRes)
    {
		case WAIT_FAILED:
			OutputDebugString(_T("SerialChannel:StopThreads: Wait failed."));
			break;
        case WAIT_OBJECT_0:
        case WAIT_OBJECT_0 + 1: 
            dwRes = WAIT_OBJECT_0;
            break;

        case WAIT_TIMEOUT:
            
            if (WaitForSingleObject((HANDLE)m_readerThread, 0) == WAIT_TIMEOUT)
                OutputDebugString(_T("SerialChannel:StopThreads: Reader Thread didn't exit."));

            if (WaitForSingleObject((HANDLE)m_readerThread, 0) == WAIT_TIMEOUT)
                OutputDebugString(_T("SerialChannel:StopThreads: Writer Thread didn't exit."));

            break;

        default:

            break;
    }

    //
    // reset thread exit event here
    //
    m_shutdownEvent.ResetEvent();

    return dwRes;
}

CString CSerialChannel::PortName() const
{
	return CString();
}

void CSerialChannel::SetPortName(const CString& value)
{
	m_portName = value;
}

BOOL CSerialChannel::IsOpen() const
{
	return m_serialPort.IsOpen();
}

int CSerialChannel::ReadTimeout()
{
	COMMTIMEOUTS timeouts;
	m_serialPort.GetTimeouts(timeouts);
	return timeouts.ReadIntervalTimeout;
}

void CSerialChannel::SetReadTimeout(int timeout)
{
	COMMTIMEOUTS timeouts;
	timeouts.ReadIntervalTimeout = timeout;
	m_serialPort.SetTimeouts(timeouts);
}


void CSerialChannel::Open()
{
	if(IsOpen())
		return;


	m_syncObject.Lock();
	CString port =  m_portName;
	port = CString("\\\\.\\") + port;
	m_serialPort.Open(port, 9600, CSerialPort::NoParity, 8, CSerialPort::OneStopBit, CSerialPort::NoFlowControl, TRUE);
	
	m_serialPort.Setup(4096, 4096);

	m_serialPort.SetMask(EV_RXCHAR);
	// flush the port
	m_serialPort.Purge(PURGE_RXCLEAR | PURGE_TXCLEAR | PURGE_RXABORT | PURGE_TXABORT);


	m_syncObject.Unlock();

	StartThreads();

}

void CSerialChannel::Close()
{
	if(IsOpen())
	{
		if (StopThreads() != WAIT_OBJECT_0)
			OutputDebugString(_T("SerialChannel:Error waiting for threads.\r\n"));

		m_syncObject.Lock();
		m_serialPort.Close();
		m_syncObject.Unlock();
	}
}

void CSerialChannel::WriteLine(const CString& text)
{	
	m_szWriteBuffer.Add(text);
	m_writeEvent.SetEvent();
}

CString CSerialChannel::TakeWriteBuffer()
{
	CString buffer;
	if(m_szWriteBuffer.GetCount() > 0)
	{
		buffer = m_szWriteBuffer.GetAt(0);
		m_szWriteBuffer.RemoveAt(0);
	}
	return buffer;
}

CString CSerialChannel::ReadExisting()
{
	CString buffer;
	while(m_szReadBuffer.GetCount() != 0)
	{		
		buffer.Append(m_szReadBuffer.GetAt(0));
		m_szReadBuffer.RemoveAt(0);
	}

	return buffer;
}

UINT CSerialChannel::ReaderProc(LPVOID pParam)
{
	CSerialChannel *channel = (CSerialChannel*)pParam;

	CSerialPort* port = &channel->m_serialPort;
	
    OVERLAPPED	overlapped = {0};  // overlapped structure

	DWORD dwBytesRead = 0; 
	DWORD dwEvent = 0;
	DWORD dwMask = 0;
	BOOL isRunning = TRUE;
	BOOL waitForEvent = FALSE;
	CStringA readBuffer;
	DWORD dwBytesWaiting = 0;

	CEvent readEvent(FALSE, FALSE);
	overlapped.hEvent = readEvent;//CreateEvent(NULL, TRUE, FALSE, NULL);

	HANDLE hEvents[] = {channel->m_shutdownEvent, overlapped.hEvent};

	while ( isRunning )
	{
		if(!waitForEvent)
		{
			try
			{
				if(port != INVALID_HANDLE_VALUE)
				{
					channel->m_syncObject.Lock();
					dwBytesWaiting  = port->BytesWaiting();
					char* pstr = readBuffer.GetBufferSetLength(dwBytesWaiting);
					port->Read( pstr,			
						dwBytesWaiting,			
						overlapped,					
						&dwBytesRead);

					TRACE("%s", readBuffer);
					readBuffer.ReleaseBuffer();

					channel->m_syncObject.Unlock();

					channel->OnDataReceived(CString(readBuffer));
					readBuffer.Empty();
				}

			}
			catch(CSerialException* pEx)
			{
				channel->m_syncObject.Unlock();
				if (pEx->m_dwError == ERROR_IO_PENDING)
				{
					waitForEvent = TRUE;
					pEx->Delete();
				}
				else
				{
					DWORD dwError = pEx->m_dwError;
					pEx->Delete();
					CSerialPort::ThrowSerialException(dwError);
				}
			}
		}
		DWORD dwRes = WaitForMultipleObjects(2, hEvents, FALSE, INFINITE);
		switch(dwRes)
		{
		case WAIT_OBJECT_0:	// Shutdown
			{
				isRunning = FALSE;
				AfxEndThread(100);
				break;	
			}
		case WAIT_OBJECT_0 + 1: // Read	
			{
				if(waitForEvent)
				{
					port->GetMask(dwMask);
					//if (dwMask & EV_CTS)
					//	;
					//if (dwMask & EV_RXFLAG)
					//	;
					//if (dwMask & EV_BREAK)
					//	;
					//if (dwMask & EV_ERR)
					//	;
					//if (dwMask & EV_RING)
					//	;

					if (dwMask & EV_RXCHAR)
					{
						try
						{
							channel->m_syncObject.Unlock();
							port->GetOverlappedResult(overlapped, dwBytesRead, TRUE);

							TRACE("%s", readBuffer);
							readBuffer.ReleaseBuffer();
							channel->OnDataReceived(CString(readBuffer));
							readBuffer.Empty();

							channel->m_syncObject.Unlock();
						}
						catch(CSerialException* pEx)
						{
							channel->m_syncObject.Unlock();
							DWORD dwError = pEx->m_dwError;
							pEx->Delete();
							CSerialPort::ThrowSerialException(dwError);

						}
					}
					waitForEvent = FALSE;
				}
				break;
			}
		}

	}

	return 0;
}

void CSerialChannel::ReceiveChar()
{
	/*BOOL  bRead = TRUE; 
	BOOL  bResult = TRUE;
	DWORD dwError = 0;
	DWORD dwBytesRead = 0;
	DWORD dwBytesWaiting = 0;
	//unsigned char RXBuff;
	CStringA strRXBuffer;
	CStringA strBuffer;

	//m_overlapped.Offset = 0;
	//m_overlapped.OffsetHigh = 0;

	// Gain ownership of the comm port critical section.
	// This process guarantees no other part of this program 
	// is using the port object. 

	m_syncObject.Lock();
	m_serialPort.ClearError(dwError);
	dwBytesWaiting  = m_serialPort.BytesWaiting();
	m_syncObject.Unlock();
	BOOL bWaiting = FALSE;
	
	while (dwBytesWaiting != 0)
	{		
		m_syncObject.Lock();

		if(!bWaiting)
		{
			try
			{
				LPTSTR pstr = strRXBuffer.GetBufferSetLength(dwBytesWaiting);
				m_serialPort.Read( pstr,			// RX Buffer Pointer
					dwBytesWaiting,					// Read waiting bytes
					m_overlapped,					// pointer to the m_ov structure
					&dwBytesRead);					// Stores number of bytes

				strBuffer.Append(pstr);
				TRACE("%s", strRXBuffer);

				m_syncObject.Unlock();
				if(	dwBytesRead == dwBytesWaiting)
					bWaiting = FALSE;
			}
			catch(CSerialException* pEx)
			{
				m_syncObject.Unlock();
				if (pEx->m_dwError == ERROR_IO_PENDING)
				{
					//DWORD dwBytesTransferred = 0;
					//port2.GetOverlappedResult(overlapped, dwBytesTransferred, TRUE);
					//m_serialPort.GetOverlappedResult(m_overlapped,	// Overlapped structure
					//	dwBytesRead,		// Stores number of bytes read
					//	TRUE);			// Wait flag

					bWaiting = TRUE;
					pEx->Delete();
				}
				else
				{
					DWORD dwError = pEx->m_dwError;
					pEx->Delete();
					CSerialPort::ThrowSerialException(dwError);
				}
			}
		}
		
		if(bWaiting)
		{
			bWaiting = FALSE;
			
			m_syncObject.Lock();
			m_serialPort.GetOverlappedResult(m_overlapped,	// Overlapped structure
				dwBytesRead,		// Stores number of bytes read
				TRUE);			// Wait flag


			m_syncObject.Unlock();
		}

		m_syncObject.Lock();
		dwBytesWaiting  = m_serialPort.BytesWaiting();
		m_syncObject.Unlock();
	}

	TRACE("Received in port:%s", strBuffer);
	OnDataReceived(strBuffer);
*/
}


/*UINT CSerialChannel::ReaderProc2(LPVOID pParam)
{
	CSerialChannel *channel = (CSerialChannel*)pParam;

	CSerialPort* port = &channel->m_serialPort;
	
    OVERLAPPED	overlapped = {0};  // overlapped structure

	DWORD dwBytesRead = 0; 
	DWORD dwEvent = 0;
	DWORD dwMask = 0;
	BOOL isRunning = TRUE;
	BOOL waitForEvent = FALSE;
	CStringA readBuffer;
	DWORD dwBytesWaiting = 0;

	CEvent readEvent(FALSE, TRUE);

	overlapped.hEvent = readEvent;//CreateEvent(NULL, TRUE, FALSE, NULL);

	HANDLE hEvents[] = {channel->m_shutdownEvent, overlapped.hEvent};

	while ( isRunning )
	{
		if (!waitForEvent)
		{			
			try
			{
				if(port == INVALID_HANDLE_VALUE)
					continue;
					channel->m_syncObject.Lock();
					dwBytesWaiting  = port->BytesWaiting();
					LPTSTR pstr = readBuffer.GetBufferSetLength(dwBytesWaiting);
					port->Read( pstr,			
						dwBytesWaiting,			
						overlapped,					
						&dwBytesRead);

					TRACE("%s", readBuffer);
					readBuffer.ReleaseBuffer();
					
					channel->m_syncObject.Unlock();

					channel->OnDataReceived(readBuffer);
					readBuffer.Empty();

			}
			catch(CSerialException* pEx)
			{
				channel->m_syncObject.Unlock();
				if (pEx->m_dwError == ERROR_IO_PENDING)
				{
					waitForEvent = TRUE;
					pEx->Delete();
				}
				else
				{
					DWORD dwError = pEx->m_dwError;
					pEx->Delete();
					CSerialPort::ThrowSerialException(dwError);
				}
			}
		}
     
		//
		// wait for pending operations to complete
		//
		if ( waitForEvent )
		{
			DWORD dwRes = WaitForMultipleObjects(2, hEvents, FALSE, INFINITE);
			switch(dwRes)
			{
			case WAIT_OBJECT_0:	// Shutdown
				{
					isRunning = FALSE;
					break;	
				}
			case WAIT_OBJECT_0 + 1: // Read	
				{
					port->GetMask(dwMask);
					//if (dwMask & EV_CTS)
					//	;
					//if (dwMask & EV_RXFLAG)
					//	;
					//if (dwMask & EV_BREAK)
					//	;
					//if (dwMask & EV_ERR)
					//	;
					//if (dwMask & EV_RING)
					//	;
			
					if (dwMask & EV_RXCHAR)
					{
						try
						{
							channel->m_syncObject.Unlock();
							port->GetOverlappedResult(overlapped, dwBytesRead, TRUE);

							TRACE("%s", readBuffer);
							readBuffer.ReleaseBuffer();
							channel->OnDataReceived(readBuffer);
							readBuffer.Empty();

							channel->m_syncObject.Unlock();
						}
						catch(CSerialException* pEx)
						{
							channel->m_syncObject.Unlock();
							DWORD dwError = pEx->m_dwError;
							pEx->Delete();
							CSerialPort::ThrowSerialException(dwError);
							
						}
					}
					waitForEvent = FALSE;
					break;
				}
			}
		}
	}
	
	return 0;
}*/


UINT CSerialChannel::WriterProc(LPVOID pParam)
{
	CSerialChannel *channel = (CSerialChannel*)pParam;

	CSerialPort* port = &channel->m_serialPort;
	
    OVERLAPPED	overlapped = {0};  // overlapped structure
	
	DWORD dwBytesToWrite = 0;
	DWORD dwBytesWritten = 0;
	DWORD dwEvent = 0;
	DWORD dwMask = 0;
    BOOL isRunning = TRUE;
	BOOL waitForEvent = FALSE;
	CStringA writeBuffer;
				
	CEvent writeEvent(FALSE, FALSE);

	overlapped.hEvent = writeEvent;//CreateEvent(NULL, TRUE, FALSE, NULL);

	HANDLE hEvents[] = {channel->m_shutdownEvent, channel->m_writeEvent};
		
	// begin forever loop.  This loop will run as long as the thread is alive.
	while ( isRunning )
	{ 
		DWORD dwRes = WaitForMultipleObjects(2, hEvents, FALSE, INFINITE);
		
		switch(dwRes)
        {
            case WAIT_OBJECT_0: // Shutdown
                isRunning = FALSE;
                break;
            
            case WAIT_OBJECT_0 + 1: // Write

				if(!waitForEvent)
				{
					writeBuffer = channel->TakeWriteBuffer();
					
					try
					{
						CStringA ascii(writeBuffer);
						dwBytesToWrite = ascii.GetLength();
						if(dwBytesToWrite > 0)
						{							
							channel->m_syncObject.Lock();
							port->Write(ascii, 
								dwBytesToWrite, 
								overlapped, 
								&dwBytesWritten);
							
							channel->m_syncObject.Unlock();
							channel->OnDataTransmitted(CString(ascii));
						}
					}
					catch(CSerialException* pEx)
					{
						channel->m_syncObject.Unlock();
						if (pEx->m_dwError == ERROR_IO_PENDING)//ERROR_INVALID_PARAMETER
						{
							waitForEvent = TRUE;
							pEx->Delete();
						}
						else
						{
							DWORD dwError = pEx->m_dwError;
							pEx->Delete();
							CSerialPort::ThrowSerialException(dwError);
						}
					}
				}

				if(waitForEvent)
				{
					DWORD dwRes = WaitForSingleObject(writeEvent, INFINITE);
					switch(dwRes)
					{
					case WAIT_OBJECT_0:
						{
							try
							{
								channel->m_syncObject.Lock();
								port->GetOverlappedResult(overlapped, dwBytesWritten, TRUE);								
								channel->m_syncObject.Unlock();
								waitForEvent = FALSE;
								if(dwBytesWritten == dwBytesToWrite)
								{
									channel->OnDataTransmitted(CString(writeBuffer));
								}
								else
								{
									TRACE("Data written does not match data to write");
									ASSERT(TRUE);
								}
							}
							catch(CSerialException* pEx)
							{
								channel->m_syncObject.Unlock();
								DWORD dwError = pEx->m_dwError;
								pEx->Delete();
								CSerialPort::ThrowSerialException(dwError);

							}
							break;
						}
					default:
						ASSERT(TRUE);
					}

				}
				

                break;
        }
	}


	return 0;
}


void CSerialChannel::TransmitChar()
{
	/*CStringA strBuffer;
	DWORD dwBytesTransferred = 0;
	
	m_writeEvent.ResetEvent();

	while(m_szWriteBuffer.GetCount() != 0)
	{
		strBuffer = m_szWriteBuffer.GetAt(0);
		m_szWriteBuffer.RemoveAt(0);
		// Gain ownership of the critical section
		m_syncObject.Lock();

		m_overlapped.Offset = 0;
		m_overlapped.OffsetHigh = 0;

		// Clear buffer
		m_serialPort.Purge(PURGE_RXCLEAR | PURGE_TXCLEAR | PURGE_RXABORT | PURGE_TXABORT);

		try
		{
			m_serialPort.Write(strBuffer, 
				strBuffer.GetLength(), 
				m_overlapped, 
				&dwBytesTransferred);

			m_syncObject.Unlock();
		}
		catch(CSerialException* pEx)
		{
			if (pEx->m_dwError == ERROR_IO_PENDING)//ERROR_INVALID_PARAMETER
			{
				DWORD dwBytesTransferred = 0;
				m_serialPort.GetOverlappedResult(m_overlapped, dwBytesTransferred, TRUE);
				pEx->Delete();

				m_syncObject.Unlock();
			}
			else
			{

				m_syncObject.Unlock();
				DWORD dwError = pEx->m_dwError;
				pEx->Delete();
				CSerialPort::ThrowSerialException(dwError);
			}
		}

		OnDataTransmitted(strBuffer);
	}*/
}


void CSerialChannel::OnDataReceived(const CString& buffer)
{
	if(!buffer.IsEmpty())
	{
		m_szReadBuffer.Add(buffer);
		NotifyCaller(DataRead, buffer);
	}
}

void CSerialChannel::OnDataTransmitted(const CString& buffer)
{
	if(!buffer.IsEmpty())
	{
		NotifyCaller(DataWritten, buffer);
	}
}
