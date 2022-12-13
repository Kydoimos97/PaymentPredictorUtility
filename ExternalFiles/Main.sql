-- Create Views

--  Get Account Age
CREATE OR ALTER VIEW [dbo].[vwX_account_age] as
    select acctrefno,
        MAX(Contract_Age) as current_contract_age
    FROM Warehouse.dbo.tblRollRate
    group by acctrefno;
go

-- Aggregations
--  Transaction History
CREATE OR ALTER VIEW  [dbo].[vwX_transaction_history_aggregated] as
    SELECT *
    FROM(
        SELECT
            acctrefno,
            MAX(payment_number) as payments_made,
            SUM(IIF(transaction_description like '%reverse%' and transaction_description like '%payment%', 1, 0)) as reverse_payments,
            AVG(CASE WHEN transaction_code = 124 THEN transaction_amount END) as monthly_payment,
            COALESCE(AVG(CASE WHEN transaction_code = 150 THEN transaction_amount END), 0) as late_fee,
            SUM(CASE
                WHEN transaction_description like 'Late Fee' THEN 1
                WHEN transaction_description like 'Reverse Late Fee' THEN -1
                ELSE 0 END) as late_fee_count,
            SUM(CASE
                WHEN transaction_description like 'Late Fee Payment' THEN 1
                WHEN transaction_description like 'Reverse Late Fee Payment' THEN -1
                ELSE 0 END) as late_fees_payment_count,
            SUM(CASE
                WHEN transaction_description like 'NSF Fee' THEN 1
                WHEN transaction_description like 'Reverse NSF Fee' then -1
                ELSE 0 END) as nsf_fees_count,
            SUM(CASE
                WHEN transaction_description like 'NSF Fee Payment' THEN 1
                WHEN transaction_description like 'Reverse NSF Fee Payment' then -1
                ELSE 0 END) as nsf_payment_count,
            SUM(IIF(payment_method_no = 10 and transaction_description like '%payment%' and
                    transaction_description not like '%reverse%', 1, 0)) as online_payment_count
        FROM Nortridge.dbo.loanacct_trans_history
        GROUP BY acctrefno) as lth_temp
--WHERE payments_made - reverse_payments > 2 [Can use this to only select a certain group of people
;
go

-- Transaction Stats
CREATE OR ALTER VIEW [dbo].[vwX_loanacct_statistical_aggregated] as
    SELECT *
        FROM Nortridge.dbo.loanacct_statistics
        WHERE year_number = 0 and month_number = 0
go

-- Rollrate
create or alter view [dbo].vwX_rollrate_current as
SELECT trr.acctrefno, Loan_Number, RunDate, Collections_Status, Collections_Status_Previous, Contract_Portfolio, Contract_dtmContract, Contract_OriginalAmtFinanced, Contract_Age, Contract_BKType, Contract_CreditTier, CurrentLate_Parent, CurrentLate, CurrentLate_Bucket, PreviousLate, PreviousLate_Bucket, Current_Principal, Previous_Principal, bitShadow, Default_dtmRepo, Default_dtmDateOfSale, Default_SalePrice, bitInventory, Default_MetalEstimate, ExtensionCount, Contract_LoanSatus, Contract_dtmBooking, bitBooked, Collections_Collector, Collections_Collector_Manager, Collections_LateBucket, Parent_Default_fltChargeoffProvision, Contract_DealerState, Current_Principal_GL, Previous_Principal_GL, CurrentLate_GL, CurrentLate_Bucket_GL, Current_APR, TERM, CurrentAPR_BY_CurrentPrincipalGL, Term_BY_CurrentPrincipalGL, current_contract_age
FROM Warehouse.dbo.tblRollRate as trr
LEFT OUTER JOIN Warehouse.dbo.vwX_account_age as vxaa
    on trr.acctrefno = vxaa.acctrefno
 WHERE Contract_Age = current_contract_age
go



-- Create Main Table
CREATE OR ALTER VIEW  [dbo].[vwX_statistical_main]
as

    SELECT
    -- Reference Columns
        cif.cifnumber as cifmain,
        la.acctrefno,
        la.cifno as cif_depreciated,

    -- Customer Information
        cif.cifnumber,
        cif.state,
        cif.city,
        cif.zip,
        LEFT(cif.zip, 5) as trunc_zip,
        cif.entity,
        IIF(cif.disable_NCOA = 0, 1, 0) as disable_NCOA,
        IIF(cif.deceased_date IS NULL, 0, 1) as app_deceased_flag,
        IIF(cif.deceased_date2 IS NULL, 0, 1) as coapp_deceased_flag,
        IIF(cif.contact_method = -1, 1, 0) as contact_method,

    -- Loan Account Information
        la.loan_class1_no,
        la.loan_class2_no,
        la.risk_rating_no,
        la.status_code_no,
        la.collection_officer_no,

        -- Dates
        la.open_date,
        la.curr_date,
        la.open_maturity_date,
        la.curr_maturity_date,
        la.last_activity_date,
        la.payoff_date,
        IIF(la.payoff_date is not NULL, 1, IIF(la.next_billing_date < la.payoff_date, 0 ,1)) as loan_paid_off,
        la.next_accrual_cutoff,
        la.next_billing_date,
        DATEDIFF(day, DATEADD(month, -1, la.next_billing_date), la.last_activity_date) AS payment_days_late,
        la.days_in_current_period,
        la.days_past_due,

        -- Current Loan
        la.current_note_amount,
        la.current_principal_balance,
        la.current_fees_balance,
        la.current_late_charge_balance,
        la.current_payoff_balance,
        la.current_perdiem,
        la.current_impound_balance,

        --Interest Rate
        la.current_interest_rate,
        la.current_interest_balance,
        la.starting_interest_rate,
        la.starting_interest_rate - la.current_interest_rate as interest_difference,
        la.last_rate_change_date,

        -- Totals
        la.total_current_due_balance,
        la.total_past_due_balance,
        la.times_extended,
        la.times_renewed,

    -- Collateral Information
        cv.year as vehicle_year,
        cv.make as vehicle_make,
        cv.model as vehicle_model,
        cv.style as vehicle_style,
        cv.original_miles,
        cv.original_value,
        cv.current_value,
        cv.original_value - cv.current_value as value_depreciation,
        cv.valuation_method,
        cv.purchase_price,
        cv.purchase_price - cv.original_value as value_price_diff,

    -- Payment Behavior BadData
        vxtha.payments_made,
        vxtha.reverse_payments,
        vxtha.monthly_payment,
        vxtha.late_fee,
        vxtha.late_fee_count,
        vxtha.late_fees_payment_count,
        vxtha.nsf_fees_count,
        vxtha.nsf_payment_count,
        vxtha.online_payment_count,

    -- Trans action history data
        vxlsa.loan_balance_total,
        vxlsa.loan_balance_high,
        vxlsa.loan_balance_average,
        vxlsa.loan_balance_low,
        vxlsa.loan_balance_days,

        -- Paid, earned, waived
        vxlsa.principal_advanced,
        vxlsa.principal_advanced_average,
        vxlsa.principal_paid,
        vxlsa.interest_earned,
        vxlsa.interest_paid,
        vxlsa.interest_waived,
        vxlsa.late_charges_earned,
        vxlsa.late_charges_paid,
        vxlsa.late_charges_waived,
        vxlsa.fees_earned,
        vxlsa.fees_paid,
        vxlsa.fees_waived,

        --Late Buckets
        vxlsa.days_late_10,
        vxlsa.days_late_30,
        vxlsa.days_late_60,
        vxlsa.days_late_90,
        vxlsa.days_late_over,

    -- Defi BadData

        -- Decision
        IIF(tdd.strIntialDecisionUW like 'System', 1, 0) as app_system_approved,
        -- Auto Approval [Doesn't exist yet]
        IIF(tdd.strInitialDecision = tdd.strMostRecentDecision, 0, 1) as decision_changed,
        CAST(tdd.strTier AS float) as tier,

        -- Economic BadData Apps
        tdd.AmountFinanced,
        tdd.Discount,
        (tdd.discount/tdd.AmountFinanced * 100) as percentage_discount,
        tdd.Reserve,
        tdd.PTI as payment_to_income,
        tdd.APR as annual_percentage_rate,
        tdd.Payment,
        IIF(tdd.Payment = monthly_payment, 1, 0) as payment_calc_test,
        tdd.BuyRate,
        tdd.term,
        tdd.income,
        tdd.BookValue,
        tdd.ApplicantIncome,
        tdd.CoApplicantIncome,
        (tdd.ApplicantIncome/IIF(tdd.income = 0,NULL, tdd.income) * 100)                                                                                        as percentage_app_income,
        (tdd.CoApplicantIncome/IIF(tdd.income = 0,NULL, tdd.income) * 100)                                                                                      as percentage_coapp_income,
        tdd.fico,
        tdd.CoAppFICO,
        (IIF(tdd.FICO is Null, 0, tdd.FICO) + IIF(tdd.CoAppFICO is Null, 0, tdd.CoAppFICO))                                                                     as AppFICO,
        IIF(((tdd.AppFICO - tdd.FICO) / IIF(tdd.FICO = 0,NULL, tdd.FICO) * 100) = NULL, 0, ((tdd.AppFICO - tdd.FICO) / IIF(tdd.FICO = 0,NULL, tdd.FICO) * 100)) as percentage_diff_app_fico,
        tdd.CashDown,

        -- Socio BadData App
        tdd.HousingType,

        -- Current Rollrate
        vxrc.Collections_Status                                                                                                                                 as current_collection_status,
        vxrc.Collections_Status_Previous                                                                                                                        as previous_collection_status,
        vxrc.CurrentLate_Bucket                                                                                                                                 as current_late_bucket,
        vxrc.PreviousLate_Bucket as previous_late_bucket,
        vxrc.Collections_LateBucket as current_collections_late_bucket,
        vxrc.CurrentLate_GL as current_days_late,
        vxrc.Contract_BKType,
        vxrc.Contract_CreditTier




    -- Joins
    FROM Nortridge.dbo.loanacct as la RIGHT OUTER JOIN (
        Select * FROM Nortridge.dbo.cif as cif
            WHERE cifnumber > 600425014 and cifnumber IN (
            -- Sub query to filter out non 600 and edge cases leaves 6792 cases
                SELECT
                    cif.cifnumber
                FROM Nortridge.dbo.cif as cif
                WHERE (cifnumber not like '^[0-9]+$') and cifnumber like '600%'))
        AS cif
        ON LA.loan_number = cif.cifnumber
    LEFT OUTER JOIN Nortridge.dbo.loanacct_collateral_link as lcl
        ON la.acctrefno = lcl.acctrefno
    LEFT OUTER JOIN Nortridge.dbo.collateral_insurance as ci
        on lcl.collateral_id = ci.collateral_id -- Unused
    LEFT OUTER JOIN Nortridge.dbo.collateral_vehicle as cv
        on lcl.collateral_id = cv.collateral_id
    LEFT OUTER JOIN Warehouse.dbo.vwX_transaction_history_aggregated as vxtha
        on la.acctrefno = vxtha.acctrefno
    LEFt OUTER JOIN Warehouse.dbo.vwX_loanacct_statistical_aggregated as vxlsa
        on vxlsa.acctrefno = la.acctrefno
    LEFT OUTER JOIN Warehouse.dbo.vwDefiData as tdd
        on la.loan_number = tdd.Loan_number -- This might not be a good join
    LEFT OUTER JOIN Warehouse.dbo.vwX_rollrate_current as vxrc
        on la.acctrefno = vxrc.acctrefno
        ;
go

CREATE OR ALTER VIEW dbo.vwXmain_transactions AS
SELECT lth.transrefno,
       lth.transaction_code,
       lth.payment_number,
       vXsm.*,
       lth.transaction_date,
       lth.transaction_amount,
       lth.payment_method_no
FROM vwX_statistical_main as vXsm
         RIGHT OUTER JOIN Nortridge.dbo.loanacct_trans_history as lth
                          on vXsm.acctrefno = lth.acctrefno;
go


-- Create final output table
DROP TABLE IF EXISTS Warehouse.dbo.tblXmain_transactions;
go

SELECT *
INTO [dbo].[tblXmain_transactions]
FROM Warehouse.dbo.vwXmain_transactions
    WHERE acctrefno is not NULL and payment_number > 0 and status_code_no != 0;
go

-- Can get more out of transaction history aggregated but need data dictionary for (status codes and payment type code)

-- ACH is automatic payment

-- Columns at the end of tbldefidata are for autoapproval



